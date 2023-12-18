#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  July 19  2023

@author: ChenYu Xue

Contact: xuechenyu@stu.xjtu.edu.cn
School of Mathematics and Statistics
Xi'an Jiaotong University
"""

import ast
import torch
import torch.nn as nn
import torchvision  
import scipy.io as sio
import numpy as np
import pandas as pd  
import glob 
import os
import time        
import re 
from typing import Callable     
from scipy.io import savemat
from layers import *
from sphericalunet.models.layers import *
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score   
from torch.utils.data import WeightedRandomSampler,BatchSampler  
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable  

from model import muilt_view_10242, muilt_view_10242_ori, muilt_view_40962, muilt_view_40962_ori,muilt_view_encode_10242
from tensorboardX import SummaryWriter  
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized 

writer = SummaryWriter('log')

""" hyper-parameters """
random_seed = 42
cuda = torch.device('cuda:0')
batch_size = 40

model_name = 'muilt_view_10242_ori' 
up_layer = 'upsample_interpolation' 
in_channels = 3 
out_channels = 2
learning_rate =0.001 
momentum = 0.99
weight_decays = 0.0001  
fold = 1 
alpha = 0.2
bate = 0.2
gamma = 0.1
delta = 0.5
margin = 1
point_num = 10242
low_point = 162
################################################################


#Balance data in each batch
class MyBatchSampler(Sampler[List[int]]):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, sampler: Sampler[int], num_label:int, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = np.array(sampler.get_label())  
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.label_sample = [[]]*num_label  
        self.num_label = num_label
        for i in range(num_label):
            self.label_sample[i] = np.where(self.sampler==i)[0]
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        batch_num = (len(self.sampler) // self.batch_size)
        save = [0]*self.num_label  
        weight = np.array([len(self.label_sample[i]) for i in range(self.num_label)])
        for i in range(batch_num):
            for j in range(self.num_label):
                if len(self.label_sample[j]) > 0:  
                    batch.append(self.label_sample[j][save[j]])
                    save[j] += 1
                    weight[j] -= 1
            for j in range(self.num_label,self.batch_size):
                now_weight = weight.copy()
                for k in range(self.num_label):
                    if now_weight[k] < batch_num - i:
                        now_weight[k] = 0
                now_weight = now_weight / sum(now_weight)  
                label = np.random.choice(self.num_label, 1, p=now_weight)[0]
                batch.append(self.label_sample[label][save[label]])
                save[label] += 1
                weight[label] -= 1
        return iter(batch)
                
    
    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return (len(self.sampler) // self.batch_size) * self.batch_size  
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """
    def __init__(self, dataset, num_samples: int=None, indices:int = None, callback_get_label: Callable = None):

        self.indices = list(range(len(dataset))) if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label  
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples  
        # distribution of classes in the dataset
        df = pd.DataFrame() 
        df["label"] = self._get_labels(dataset)  
        df.index = self.indices  
        df = df.sort_index()  
        label_to_count = df["label"].value_counts()  
        weights = 1.0 / label_to_count[df["label"]]  
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]  
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_label()
        else:
            raise NotImplementedError
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


#val set dataset
class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, point=10242):

        self.files = root1 
        self.classes_label = []
        self.point = int(point)

        for file in self.files:
        
            file = val_data + '/' + file
            class_id = 0
            label = sio.loadmat(file)['ori_label']  
            if label == 0:
                class_id = 0
            elif label == 1:
                class_id = 1
            else:
                assert 1 == 2
            self.classes_label.append(class_id)


    def __getitem__(self, index):
        file = self.files[index]
        data_file = val_data + '/' + file
        data = sio.loadmat(data_file)
        ori_label = data['ori_label'] # 0/1 label
        label=data['label']  #one-hot label
        
        feats_right = data['feats_right'] # right hemisphere
        feats_left = data['feats_left'] # left hemisphere

        return feats_left.astype(np.float32), feats_right.astype(np.float32), label.astype(np.int32), ori_label.astype(np.int32)

    def __len__(self):
        return len(self.files)

    def get_label(self):
        return self.classes_label


#train set dataset

class BrainSphere_train(torch.utils.data.Dataset):  

    def __init__(self, root, point=10242):

        self.files = root
        self.point = int(point)
        #balance different classes
        self.classes_label = []
        for file in self.files:
            file = train_data + '/' + file
            class_id = 0
            label = sio.loadmat(file)['ori_label']
            if label == 0:
                class_id = 0
            elif label == 1:
                class_id = 1
            else:
                assert 1 == 2
            self.classes_label.append(class_id)



    def __getitem__(self, index):
        file = self.files[index]
        data_file = train_data + '/' + file
        data = sio.loadmat(data_file)
        ori_label = data['ori_label'] # 0/1 label
    
        label=data['label']   #one-hot label
        feats_right = data['feats_right'] # right hemisphere

        feats_left = data['feats_left'] # left hemisphere


        return feats_left.astype(np.float32), feats_right.astype(np.float32), label.astype(np.int32), ori_label.astype(np.int32)
    
    def __len__(self):
        return len(self.files)
    
    def get_label(self):
        return self.classes_label

#data address
train_data = 'D:/data/DHCP/train_data'
val_data = 'D:/data/DHCP/val_data'


lists_val = os.listdir(val_data)
lists_train = os.listdir(train_data)
fold1 = lists_train
random_order = list(range(len(fold1)))
np.random.seed(random_seed)
np.random.shuffle(random_order) 
fold1 = [fold1[i] for i in random_order]
fold2 = lists_val


#choose model
if model_name == 'muilt_view_10242':
    model = muilt_view_10242(in_ch=in_channels, out_ch=out_channels)
    train_dataset = BrainSphere_train(fold1)
    val_dataset = BrainSphere(fold2)

elif model_name == 'muilt_view_encode_10242':
    model = muilt_view_40962(in_ch=in_channels, out_ch=out_channels)
    train_dataset = BrainSphere_train(fold1)
    val_dataset = BrainSphere(fold2)


elif model_name == 'muilt_view_40962_ori':
    model = muilt_view_40962_ori(in_ch=in_channels, out_ch=out_channels)
    train_dataset = BrainSphere_train(fold1)
    val_dataset = BrainSphere(fold2)

elif model_name == 'muilt_view_10242_ori':
    model = muilt_view_10242_ori(in_ch=in_channels, out_ch=out_channels)
    train_dataset = BrainSphere_train(fold1)
    val_dataset = BrainSphere(fold2)





sampler_train = MyBatchSampler(train_dataset, num_label=2, batch_size=batch_size, drop_last=True)
sampler_val = ImbalancedDatasetSampler(val_dataset)

train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, batch_size=batch_size, shuffle=False ,pin_memory=True, drop_last=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=sampler_val, batch_size=1, shuffle=False, pin_memory=True)

#Define learning parameters
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))


model.to(cuda)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decays)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=20, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=1e-6)


#Define upsample index
upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index_ori()
softmax = nn.Softmax(dim=0)
up_attention_642 = up_layer_norm(1, 1, upconv_top_index_642, upconv_down_index_642)
up_attention_2562 = up_layer_norm(1, 1, upconv_top_index_2562, upconv_down_index_2562)
up_attention_10242 = up_layer_norm(1, 1, upconv_top_index_10242, upconv_down_index_10242)
up_attention_40962 = up_layer_norm(1, 1, upconv_top_index_40962, upconv_down_index_40962)

#Stability-Aware Regularization
def reconstruction(low_map, up_map):
    low_map = low_map.unsqueeze(1)
    low_map = up_attention_642(low_map) 
    low_map = up_attention_2562(low_map)
    low_map = up_attention_10242(low_map)
    low_map = low_map.squeeze(1).cuda(cuda)
    loss = (torch.mean((low_map - up_map) ** 2)) ** (1/2)
    return loss

#Fidelity-Aware Contrastive Learning
def contrastive_loss(map, map_minus, ori_label):
    batch, channel, height = map.size()
    pre_where = torch.where(ori_label == 0)[0]
    nc_where = torch.where(ori_label == 1)[0]
    if pre_where.size()[0] != 0:
        pre = map[pre_where].cuda(cuda)
        pre_minus = map_minus[pre_where].cuda(cuda)
    else:
        pre = torch.zeros((1, channel, height)).cuda(cuda)
        pre_minus = torch.zeros((1, channel, height)).cuda(cuda)
        print('error, no pre')
    if nc_where.size()[0] != 0:
        nc = map[nc_where].cuda(cuda)
    else:
        nc = torch.zeros((1, channel, height)).cuda(cuda)
        print('error, no nc')

    pre = pre.mean(dim=0)
    pre = pre.mean(dim=1)
    pre = F.normalize(pre, dim=0)
    pre_minus = pre_minus.mean(dim=0)
    pre_minus = pre_minus.mean(dim=1)
    pre_minus = F.normalize(pre_minus, dim=0)
    nc = nc.mean(dim=0)
    nc = nc.mean(dim=1)
    nc = F.normalize(nc, dim=0)


    d = (torch.mean((nc - pre_minus) ** 2)) ** (1/2)
    loss_same = ((d ** 2) / 2).mean()
 
    p = max((margin - (torch.mean((pre - pre_minus) ** 2)) ** (1/2)).item(), 0)
  
    q = max((margin - (torch.mean((nc - pre) ** 2)) ** (1/2)).item(), 0)
  
  
    loss_minus_1 = ((p ** 2) / 2)
    loss_minus_2 = ((q ** 2) / 2)
   
    loss = loss_same + 0.5 * loss_minus_1 + 0.5 * loss_minus_2

    return loss

#Sparsity-Aware Regularization
def entropy_loss(attention, ori_label):
    pre_where = torch.where(ori_label == 0)[0]
    x = attention[pre_where].cuda(cuda)
    x = x.mean(dim=0)
    x = softmax(x)
    entropy_pre = - (x * torch.log2(x)).sum()
    loss_pre = entropy_pre
    nc_where = torch.where(ori_label == 1)[0]
    y = attention[nc_where].cuda(cuda)
    y = y.mean(dim=0)
    y = softmax(y)
    entropy_nc = - (y * torch.log2(y)).sum()
    loss_nc = 1 - entropy_nc  #？
    loss = (loss_pre + loss_nc)/2
    return loss


def Sparsity_loss(attention, ori_label):
    nc_where = torch.where(ori_label == 1)[0]
    y = attention[nc_where].cuda(cuda)
    y = y.mean(dim=0)
    entropy_nc = 0.5 * abs(y).mean() + 0.5 * ((y ** 2).mean()) ** (1 / 2)
    loss = 1 - entropy_nc
    return loss


#train step
def train_step(data1, data2 ,target, ori_label, epoch,True_num):
    model.train()

    prediction, map, map_minus, attention, map_up, map_minus_up, attention_up = model(data1, data2)
    
    loss_2 = contrastive_loss(map, map_minus, ori_label)
    loss_3 = contrastive_loss(map_up, map_minus_up, ori_label)
    target = target.squeeze()
    target = target.float()
    loss_1 = criterion(prediction, target)

    loss_4 = reconstruction(attention[:, low_point:], attention_up[:, point_num:]) + reconstruction(attention[:, :low_point], attention_up[:, :point_num])
    loss_5 = entropy_loss(attention_up[:, point_num:], ori_label) + entropy_loss(attention_up[:, :point_num], ori_label)
    
    if epoch < 100:
        gamma = 0.1
    else:
        gamma = 0.001
    loss = loss_1 + alpha * loss_2 + bate * loss_3 + gamma * loss_4 + delta * loss_5

    pred = prediction.max(1)[1] 
    target = target.max(1)[1] 
    arr = pred - target
    k = (arr == 0).sum()
    True_num = k + True_num
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    return loss.item(), loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(),True_num

max_acc = 0
max_auc = 0
acc = 0.5
train_arr = 0
train_dice = [0, 0, 0, 0, 0]
sub_num = len(train_dataloader) * batch_size 
save_num = 1

for epoch in range(200):
    
    #Learning rate decay
    scheduler.step(np.mean(train_dice)) 
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    #run time statistics
    True_num = 0
    since = time.time()
    
    #train
    for batch_idx, (data1, data2, target, ori_label) in enumerate(train_dataloader):

        #input data
        data1, data2, target, ori_label = data1.cuda(cuda), data2.cuda(cuda), target.cuda(
            cuda), ori_label.cuda(cuda)
        #loss function
        loss, loss_1, loss_2, loss_3, loss_4, loss_5, True_num = train_step(data1, data2, target, ori_label, epoch, True_num)
        print("[{}:{}/{}]  LOSS={:.4}  LOSS_1={:.4}  LOSS_2={:.4}  LOSS_3={:.4}  LOSS_4={:.4}  LOSS_5={:.4}  ".format(epoch,
                             batch_idx, len(train_dataloader), loss, loss_1, loss_2, loss_3, loss_4, loss_5))

        writer.add_scalar('Train/Loss', loss, epoch * len(train_dataloader) + batch_idx)
    train_arr = (True_num/sub_num).item()
    print("Train_arr={:.4}".format(train_arr))

    True_num = 0

    pred_list = []
    label_list = []
    #val step
    for batch_idx, (data1, data2, target, ori_label) in enumerate(val_dataloader):
        model.eval()
        data1, data2, target, ori_label = data1.cuda(cuda), data2.cuda(cuda), target.cuda(cuda), ori_label.cuda(cuda)

        prediction, map, map_minus, attention, map_up, map_minus_up, attention_up = model(data1, data2)
        target = target.squeeze()
        target = target.float()
        pred_list.append(prediction.cpu().detach().numpy())
        label_list.append(ori_label.cpu().detach().numpy())
        pred = prediction.max(1)[1]
        target = target.max(0)[1]
        arr = pred - target
        k = (arr == 0).sum()
        True_num = True_num + k

    acc = True_num/len(fold2)

 

    pred_list = np.squeeze(np.array(pred_list))
    label_list = np.squeeze(np.array(label_list))
    pred_list = pred_list[:, 1]
    auc = roc_auc_score(label_list, pred_list)
    print("Val_acc={:.4}, Val_auc={:.4}".format(acc.item(), auc))

    # save model with max acc or auc
    if (acc >= max_acc) and (auc >= max_auc):
        max_acc = acc
        max_acc = max_acc.item()
        max_auc = auc
        save_epoch = epoch
        #save model
        print('save max acc or auc model')
        torch.save(model.state_dict(), os.path.join('train_models', model_name + '_'+ str(save_epoch) + '_max_acc' +".pkl"))
        
        save_num += 1
    print("Val_max_auc={:.4}, Val_max_arr={:.4}, save_epoch={}, save_num={}".format(max_auc, max_acc, save_epoch, save_num))
    acc = acc.item()
    train_dice[epoch % 5] = train_arr
    #save model
    print("last five train Dice: ", train_dice)
    torch.save(model.state_dict(), os.path.join('train_models', model_name + '_'+ str(fold) + ".pkl"))
  
    writer.add_scalars('Train/Val', {'val_acc': acc, 'val_auc': auc, 'train': train_arr}, epoch)
    time_elapsed = time.time() - since
    print('All complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
writer.close()
