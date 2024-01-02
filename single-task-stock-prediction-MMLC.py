#!/usr/bin/env python
# coding: utf-8

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pandas as pd
import cv2
import os
import argparse

import logging
from logger import get_logger

from meta_modelsXY import MetaNet
from stockset.resnet import resnet32
from stockset.utilsour import DataIterator

from mlc_utils import clone_parameters, tocuda, DummyScheduler
from tqdm import tqdm
from collections import deque
import copy
import sys
import pickle
from mlc_utils import soft_cross_entropy as soft_loss_f

from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True




setup_seed(6)



# TODO: Define transforms for the training data and testing data
trainx_transforms = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

trainy_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])#,transforms.Grayscale()

#测试时，不能改变图像（但是需要以同一方式标准化）。因此，在验证/测试图像时，通常只能调整大小和裁剪图像。
test_transforms = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])



def Label(path_num,window_size,num):
    a = ['train','valid','test']
    path_label ="./dataset/pre"+str(window_size)+"/"+str(path_num)+"/" + 'label_{}.csv'.format(a[num])
    label = pd.read_csv(path_label,index_col=0)
    path_clean ="./dataset/pre"+str(window_size)+"/"+str(path_num)+"/" + 'cleanlabel_{}.csv'.format(a[num])
    clean_label = pd.read_csv(path_clean)
    return label,clean_label




def load_images(path_num,window_size,label,num):
    a = ['TrainX','ValidX','TestX']
    b = ['TrainY','ValidY','TestY']
    c = ['picname_train','picname_valid','picname_test']
    d = ['label_train','label_valid','label_test']
    path_picX = "./dataset/pre"+str(window_size)+"/"+ str(path_num)+"/"  + "gafImages/"+'{}'.format(a[num])
    path_picY = "./dataset/pre"+str(window_size)+"/"+ str(path_num)+"/"  + "yImages/"+'{}'.format(b[num])
    
    filename_all = label[c[num]]
    
    imagesX = []
    imagesY = []

    for filename in filename_all:
        imgX = cv2.imread(os.path.join(path_picX,filename))
        imgY = cv2.imread(os.path.join(path_picY,filename))
        if imgY is not None:
            imgX = trainx_transforms(imgX) #Tensor的形状是[C,H,W]，而cv2，plt，PIL形状都是[H,W,C]
            imgY = trainy_transforms(imgY)
            imagesX.append(imgX)
            imagesY.append(imgY)
    X_images = torch.tensor([item.numpy() for item in imagesX])
    Y_images = torch.tensor([item.numpy() for item in imagesY])
    all_label = torch.tensor(label[d[num]])
    return X_images,Y_images,all_label



def creat_dataset(path_num,num,window_size):
    label,clean_label = Label(path_num,window_size,num)
    if num ==0:
            X_images,Y_images,labels = load_images(path_num,window_size,label,num)
    else:
        X_images,Y_images,labels = load_images(path_num,window_size,clean_label,num)
    labels= torch.tensor(labels, dtype=torch.int32)
    
    return X_images,Y_images,labels




parser = argparse.ArgumentParser(description='MMLC Stock Training Framework')
parser.add_argument('--dataset', type=str, choices=['KDD17', 'others'], default='KDD17')
parser.add_argument('--seed', type=int, default=13) 
parser.add_argument('--data_seed', type=int, default=1)
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--every', default=100, type=int, help='Eval interval (default: 100 iters)')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=3e-4, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=3e-5, type=float, help='lr for meta net')
parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--skip', default=False, action='store_true', help='Skip link for LCN (default: False)')
parser.add_argument('--sparsemax', default=False, action='store_true', help='Use softmax instead of softmax for meta model (default: False)')
parser.add_argument('--tie', default=False, action='store_true', help='Tie label embedding to the output classifier output embedding of metanet (default: False)')
############## LOOK-AHEAD GRADIENT STEPS ##################
parser.add_argument('--gradient_steps', default=10, type=int, help='Number of look-ahead gradient steps for meta-gradient (default: 1)')

# KDD17
# Positional arguments
parser.add_argument('--data_path', default='data', type=str, help='Root for the datasets.')

# Acceleration
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
# i/o
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank (-1 for local training)')

args = parser.parse_args(args=[])



# //////////////// set logging and model outputs /////////////////
filename = '_'.join([args.dataset, str(args.epochs), str(args.seed), str(args.data_seed)])
if not os.path.isdir('logs'):
    os.mkdir('logs')
logfile = 'logs/' + filename + '.log'
logger = get_logger(logfile, args.local_rank)

if not os.path.isdir('models'):
    os.mkdir('models')
# ////////////////////////////////////////////////////////////////

logger.info(args)
logger.info('CUDA available:' + str(torch.cuda.is_available()))

# cuda set up
torch.cuda.set_device(0) # local GPU
hard_loss_f = F.cross_entropy



# //////////////////////// defining model ////////////////////////
def build_metanet(dataset, num_classes):
    # meta net
    inx_dim = 30*30*3
    hx_dim = 64 #0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
    meta_net = MetaNet(inx_dim, hx_dim, num_classes, args)

    meta_net = meta_net.cuda()

    logger.info('========== Meta model ==========')
    logger.info(meta_net)

    return meta_net

def build_mainnet(dataset, num_classes):
    # main net
    model = resnet32(num_classes)#num_classes
    main_net = model

    main_net = main_net.cuda()

    logger.info('========== Main model==========')
    logger.info(model)
    return main_net



def setup_metatraining(meta_net, exp_id=None):
    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # meta net optimizer
    optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                 weight_decay=0, #args.wdecay, # meta should have wdecay or not??
                                 amsgrad=True, eps=args.opt_eps)
    scheduler = DummyScheduler(optimizer)
    last_epoch = -1
    return meta_net,optimizer,scheduler,last_epoch



def setup_maintraining(main_net, exp_id=None):
    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # main net optimizer
    main_params = main_net.parameters() 

    if args.optimizer == 'adam':
        main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True, eps=args.opt_eps)
    elif args.optimizer == 'sgd':
        main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)

    main_schdlr = DummyScheduler(main_opt)

    return main_net, main_opt, main_schdlr



def multi_class_accuracy_precision_f1score(y_true, logit_label, num_classes):
    # calculates accuracy, weighted precision, and weighted f1-score for n-class classification for n>=3
    # note that weighted recall is the same as accuracy
    y_pred = torch.max(logit_label,1)[1]
    N = len(y_true)
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for i in range(0, N):
        confusion_matrix[y_true[i]][y_pred[i]] += 1

    sum_diagonal = 0
    
    for i in range(0, num_classes):
        sum_diagonal += confusion_matrix[i][i]

    precision = 0.0
    f1score = 0.0

    for i in range(0, num_classes):
        support = 0
        sum_column = 0

        for j in range(0, num_classes):
            support += confusion_matrix[i][j]
            sum_column += confusion_matrix[j][i]

        if support != 0:
            g = confusion_matrix[i][i] * support
            f1score += g / (support + sum_column)

            if sum_column != 0:
                precision += g / sum_column

    accuracy = sum_diagonal / N
    precision /= N
    f1score = 2 * f1score / N

    return accuracy, precision, f1score



def test(main_net, test_loader): # this could be eval or test
    # //////////////////////// evaluate method ////////////////////////
    correct = torch.zeros(1).cuda()
    nsamples = torch.zeros(1).cuda()
    

    ts_acc = torch.zeros(1).cuda()
    ts_prec = torch.zeros(1).cuda()
    ts_f1 = torch.zeros(1).cuda()
    pre_out = []
    target_label = []
    
    # forward
    main_net.eval()

    for idx, (*datax,datay,target) in enumerate(test_loader):
        datax,datay, target = tocuda(datax),tocuda(datay), tocuda(target)

        # forward
        with torch.no_grad():
            output = main_net(datax)
        
        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()
        nsamples += len(target)

        pre_out.extend(output)
        target_label.extend(target)

    test_acc = (correct / nsamples).item()
    pre_label = torch.tensor([item.cpu().numpy() for item in pre_out])
    #acc, f1, precious
    ts_accuracy, ts_precision, ts_f1score = multi_class_accuracy_precision_f1score(torch.Tensor(target_label).long(), pre_label, num_classes=3)

    return test_acc, ts_accuracy, ts_precision,ts_f1score


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans


# ============== mmlc step procedure debug with features (gradient-stopped) from main model ===========

def update_tensor_grads(params, grads):
    for l, g in zip(params, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g

#计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
def _hessian_vector_product(vector, main_net, meta_net, data_sx, data_sy,target_s,soft_loss_f, r=1e-2): # vector就是dw'Lval(w',α),即为gw_prime, model就是main_net
    R = r / _concat(vector).norm() #epsilon
    #dαLtrain(w+,α)
    for p, v in zip(main_net.parameters(), vector):
        p.data.add_(R, v) #将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    logit_s, x_s_h = main_net(data_sx,return_h=True)
    pseudo_target_s = meta_net(data_sx.detach(),data_sy, target_s)
    loss = soft_loss_f(logit_s, pseudo_target_s)

    grads_p = torch.autograd.grad(loss, main_net.parameters())

  #dαLtrain(w-,α)
    for p, v in zip(main_net.parameters(), vector):
        p.data.sub_(2*R, v) #将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    logit_s, x_s_h = main_net(data_sx,return_h=True)
    pseudo_target_s = meta_net(data_sx.detach(),data_sy, target_s)
    loss = soft_loss_f(logit_s, pseudo_target_s)
    
    grads_n = torch.autograd.grad(loss, main_net.parameters())

  #将模型的参数从w-恢复成w
    for p, v in zip(main_net.parameters(), vector):
        p.data.add_(R, v) #w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

    Hww =[(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
    return Hww



# //////////////////////// load task //////////////////////////////
innerloop=int(1)
last_step = innerloop-1
path_num=0
num_tasks = 1
num_classes =3

trX_images_all,trY_images_all,trlabels_all = creat_dataset(path_num,num=0,window_size=10)
valX_images_all,valY_images_all,vallabels_all = creat_dataset(path_num,num=1,window_size=10)
tsX_images_all,tsY_images_all,tslabels_all = creat_dataset(path_num,num=2,window_size=10)

train_data_meta = list(zip(valX_images_all,valY_images_all,vallabels_all))
train_data = list(zip(trX_images_all,trY_images_all,trlabels_all))
gold_loader = DataIterator(torch.utils.data.DataLoader(train_data_meta, batch_size=args.bs, shuffle=True, pin_memory=True))
silver_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, pin_memory=True)

test_data = list(zip(tsX_images_all,tsY_images_all,tslabels_all))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False,pin_memory=True)



df_label_test= pd.DataFrame(tslabels_all,columns=['label'])
# view countplot of manual target variabel
sns.countplot(x='label',data=df_label_test)
plt.show()


df_label_test= pd.DataFrame(vallabels_all,columns=['label'])
# view countplot of manual target variabel
sns.countplot(x='label',data=df_label_test)
plt.show()




filename = '_'.join([args.dataset, str(args.epochs), str(args.seed), str(args.data_seed)])
results = {}

# //////////////////////// build main_net and meta_net/////////////
meta_net = build_metanet(args.dataset, num_classes)
main_net = build_mainnet(args.dataset, num_classes)
# # # //////////////////////// load pretrain main_net and meta_net/////////////
PATH_meta = "meta_model.pt"
meta_net.load_state_dict(torch.load(PATH_meta)) 
# //////////////////////// train and eval model ///////////////////
exp_id = '_'.join([filename])
writer = SummaryWriter(args.logdir + '/' + exp_id)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

meta_net,meta_opt,scheduler,last_epoch = setup_metatraining(meta_net, exp_id)
main_net, main_opt, main_schdlr = setup_maintraining(main_net, exp_id)

# //////////////////////// switching on training mode ////////////////////////
meta_net.train()
main_net.train()

# set done

args.dw_prev = [0 for param in meta_net.parameters()] # 0 for previous iteration
args.steps = 0



for epoch in tqdm(range(last_epoch+1, args.epochs)):# change to epoch iteration
        logger.info('Epoch %d:' % epoch)
        
        for num, (*data_sx,data_sy, target_s) in enumerate(silver_loader):
            *data_gx, data_gy, target_g = next(gold_loader)#.next()
            data_gx, data_gy, target_g = tocuda(data_gx),tocuda(data_gy), tocuda(target_g)
            data_sx, data_sy, target_s = tocuda(data_sx),tocuda(data_sy), tocuda(target_s)
            
            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]
            for i in range(innerloop):
                logit_s, x_s_h = main_net(data_sx,return_h=True)
                pseudo_target_s = meta_net(data_sx.detach(),data_sy, target_s)
                loss_s = soft_loss_f(logit_s, pseudo_target_s)
                if i == last_step:
                    # compute gw for updating meta_net
                    logit_g = main_net(data_gx)
                    #print('target',target_g.shape)
                    loss_g = hard_loss_f(logit_g, target_g.long())
                    gw = torch.autograd.grad(loss_g, main_net.parameters())
                    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)    
                else:
                    main_opt.zero_grad()
                    grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)    
                    update_tensor_grads(main_net.parameters(), grads)
                    main_opt.step()

            f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
            # 2. set w as w'
            f_param = []
            for i, param in enumerate(main_net.parameters()):
                f_param.append(param.data.clone())
                param.data = f_params_new[i].data # use data only as f_params_new has graph
                # 3. compute d_w' L_{D}(w')

            logit_g = main_net(data_gx)
            loss_g  = hard_loss_f(logit_g, target_g.long())
            gw_prime = torch.autograd.grad(loss_g, main_net.parameters())
             # modify to w, and then do hessian calculate
            for i, param in enumerate(main_net.parameters()):
                param.data = f_param[i]
                param.grad = f_param_grads[i].data
#             main_opt.step()

            Hww = _hessian_vector_product(gw_prime, main_net, meta_net, data_sx, data_sy,target_s,soft_loss_f, r=1e-2) 
            # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2

            tmp1 = [(gw_prime[i]-Hww[i]*dparam_s[i])for i in range(len(dparam_s))]

            gw_norm2 = (_concat(gw).norm())**2
            tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]

            gamma = torch.dot(_concat(tmp1), _concat(tmp2))

            # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
            Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     

            proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

            # back prop on alphas
            meta_opt.zero_grad()
            proxy_g.backward()
            
             # accumulate discounted iterative gradient
            for i, param in enumerate(meta_net.parameters()):
                if param.grad is not None:
                    param.grad.add_(gamma * args.dw_prev[i])
                    args.dw_prev[i] = param.grad.clone()

            if (args.steps+1) % (args.gradient_steps)==0: # T steps proceeded by main_net
                meta_opt.step()
                args.dw_prev = [0 for param in meta_net.parameters()] # 0 to reset 

            # modify to w, and then do actual update main_net
            for i, param in enumerate(main_net.parameters()):
                param.data = f_param[i]
                param.grad = f_param_grads[i].data
            main_opt.step()
            
            args.steps += 1
            if num % args.every == 0:
                writer.add_scalar('train/loss_g', loss_g.item(), args.steps)
                writer.add_scalar('train/loss_s', loss_s.item(), args.steps)

                main_lr = main_schdlr.get_lr()[0]
                meta_lr = scheduler.get_lr()[0]
                writer.add_scalar('train/main_lr', main_lr, args.steps)
                writer.add_scalar('train/meta_lr', meta_lr, args.steps)
                writer.add_scalar('train/gradient_steps', args.gradient_steps, args.steps)

                logger.info('Iteration %d loss_s: %.4f\tloss_g: %.3f\tMain LR: %.8f\tMeta LR: %.8f' %( i, loss_s.item(), loss_g.item(), main_lr, meta_lr))

        # PER EPOCH PROCESSING

        # lr scheduler
        main_schdlr.step()        
        #scheduler.step()


        test_acc,ts_acc1, ts_prec,ts_f1 = test(main_net, test_loader)
        logger.info('Test acc: %.4f' % (test_acc))
        logger.info('Test acc: %.4f\tTest prec: %.4f\tTest f1: %.4f' % (ts_acc1, ts_prec,ts_f1 ))



pre_out = []
target_label = []
ts_acc = torch.zeros(1).cuda()
ts_prec = torch.zeros(1).cuda()
ts_f1 = torch.zeros(1).cuda()
for idx, (*datax,datay,target) in enumerate(test_loader):
    datax,datay, target= tocuda(datax),tocuda(datay), tocuda(target)

    # forward
    with torch.no_grad():
        output = main_net(datax)

    # accuracy
    pred = output.data.max(1)[1]
    pre_out.extend(output)
    target_label.extend(target)




pre_label = torch.tensor([item.cpu().numpy() for item in pre_out])


ts_accuracy, ts_precision, ts_f1score = multi_class_accuracy_precision_f1score(torch.Tensor(target_label).long(), pre_label, num_classes=3)
print("ts_accuracy, ts_precision, ts_f1score:",ts_accuracy, ts_precision, ts_f1score)






