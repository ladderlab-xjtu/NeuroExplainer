'''
Created on  July 19  2023

@author: ChenYu Xue

Contact: xuechenyu@stu.xjtu.edu.cn
School of Mathematics and Statistics
Xi'an Jiaotong University
'''
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Callable
import nibabel as nb
from model import muilt_view_10242, muilt_view_negative_10242, muilt_view_10242_ori  #
from layers import *
from sphericalunet.utils.vtk import read_vtk, write_vtk  #读取和写入网格数据
import matplotlib.pyplot as plt

""" hyper-parameters """
batch_size = 1
cuda = torch.device('cuda:0')
point = 10242
low_point = 162
abspath = os.path.abspath(os.path.dirname(__file__))
# surface data
template_162 = read_vtk(abspath + '/neigh_indices/sphere_162_rotated_0.vtk')
point_162 = template_162['vertices']
faces_162 = template_162['faces']
template_10242 = read_vtk(abspath + '/neigh_indices/sphere_10242_rotated_0.vtk')
point_10242 = template_10242['vertices']
faces_10242 = template_10242['faces']
surface_lh = sio.loadmat('D:/dhcp/examples/lh_surf.mat')
surface_lh_10242 = surface_lh['vertex_inflated'][:point]
surface_lh_162 = surface_lh['vertex_inflated'][:low_point]
surface_rh = sio.loadmat('D:/dhcp/examples/rh_surf.mat')
surface_rh_10242 = surface_rh['vertex_inflated'][:point]
surface_rh_162 = surface_rh['vertex_inflated'][:low_point]

# generate result 
def inference(data1, data2, model, point=10242):
    data1 = torch.from_numpy(data1)
    data1 = data1.to(device)
    data1 = data1.unsqueeze(0)
    data2 = torch.from_numpy(data2)
    data2 = data2.to(device)
    data2 = data2.unsqueeze(0)
    with torch.no_grad():
        prediction, map, map_minus, attention, map_up, map_minus_up, attention_up = model(data1, data2)
    print(prediction)
    pred = prediction.max(1)[1]
    pred = pred.cpu().numpy()
    return pred, map, map_minus, attention, map_up, map_minus_up, attention_up

# data path
path = 'D:\dhcp'
val_path = 'D:/dhcp/val_data'
point_num = low_point
lists = os.listdir(val_path)
len_fold1 = int(500)
len_fold2 = int(700)
fold2 = lists
device = cuda

# load model
model = muilt_view_10242_ori(3, 2)
model_path = 'train_models/' + 'muilt_view_10242_ori_189_max_acc.pkl'
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=cuda), strict=False)
model.eval()
True_num = 0
nc_num = 0
pre_num = 0
arr_pre_num = 0
arr_nc_num = 0
nc_sparsity = 0
pre_sparsity = 0
for list in fold2:
    # date
    print(list)
    data = sio.loadmat(val_path + '/' + list)
    data1 = data['feats_left'][:3,:point]
    data2 = data['feats_right'][:3,:point]
    target = data['ori_label']
    # prediction
    pred, map, map_minus, attention, map_up, map_minus_up, attention_up = inference(data1, data2, model, point)
    print(pred)

    # Calculation accuracy
    arr = pred - target
    k = (arr == 0).sum()
    nc_num += (target == 1).sum()
    pre_num += (target == 0).sum()
    arr_pre_num += (arr == 1).sum()
    arr_nc_num += (arr == -1).sum()
    True_num = True_num + k

    # save coarse-grained attention map
    attention = attention.permute(1, 0).cpu().numpy()
    attention_left = attention[:point_num]
    attention_right = attention[point_num:]
    sphere_left_162 = {}
    sphere_right_162 = {}

    # left hemisphere
    sphere_left_162['vertices'] = point_162
    sphere_left_162['faces'] = faces_162
    sphere_left_162['attention'] = attention_left

    # sphere attention map
    new_path_left = ('D:/dhcp/hemi-lh-vtk-atten-162' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_left_162, new_path_left)
    sphere_left_162['vertices'] = surface_lh_162

    # surf attention map
    new_path_left = ('D:/dhcp/hemi-lh-vtk-surface-162' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_left_162, new_path_left)

    # right hemisphere
    sphere_right_162['vertices'] = point_162
    sphere_right_162['faces'] = faces_162
    sphere_right_162['attention'] = attention_right

    # sphere attention map
    new_path_right = ('D:/dhcp/hemi-rh-vtk-atten-162' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_right_162, new_path_right)
    sphere_right_162['vertices'] = surface_rh_162

    # surf attention map
    new_path_right = ('D:/dhcp/hemi-rh-vtk-surface-162' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_right_162, new_path_right)

    # save fine-grained attention map
    attention_up = attention_up.permute(1, 0).cpu().numpy()
    attention_left_up = attention_up[:point]
    attention_right_up = attention_up[point:]
    sphere_left_10242 = {}
    sphere_right_10242 = {}

    # left hemisphere
    sphere_left_10242['vertices'] = point_10242
    sphere_left_10242['faces'] = faces_10242
    sphere_left_10242['attention'] = attention_left_up

    # sphere attention map
    new_path_left = ('D:\dhcp\hemi-lh-vtk-atten-10242' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_left_10242, new_path_left)
    sphere_left_10242['vertices'] = surface_lh_10242

    # surf attention map
    new_path_left = ('D:\dhcp\hemi-lh-vtk-surface' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_left_10242, new_path_left)

    # right hemisphere
    sphere_right_10242['vertices'] = point_10242
    sphere_right_10242['faces'] = faces_10242
    sphere_right_10242['attention'] = attention_right_up

    # sphere attention map
    new_path_right = ('D:\dhcp\hemi-rh-vtk-atten-10242' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_right_10242, new_path_right)
    sphere_right_10242['vertices'] = surface_rh_10242

     # surf attention map
    new_path_left = ('D:\dhcp\hemi-rh-vtk-surface' + '/' + list).replace(".mat", '.vtk')
    write_vtk(sphere_right_10242, new_path_left)

    # True or False
    if (arr == 0).sum() == batch_size:
        print(True)
    else:
        print(False)
    if target == 0:
        a = np.where(attention_up > 0.6)
        b = len(a[0])
        c = 1 - b/(point*2)
        pre_sparsity += c
    elif target == 1:
        a = np.where(attention_up > 0.6)
        b = len(a[0])
        c = 1 - b/(point*2)
        nc_sparsity += c
# total accuracy
arr = True_num / len(fold2)
arr_nc = (nc_num - arr_nc_num)/nc_num
arr_pre = (pre_num - arr_pre_num)/pre_num
print("arr={:.4} arr_nc={:.4} arr_pre={:.4}".format(arr, arr_nc, arr_pre))
