'''
Created on  July 19  2023

@author: ChenYu Xue

Contact: xuechenyu@stu.xjtu.edu.cn
School of Mathematics and Statistics
Xi'an Jiaotong University
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sphericalunet.models.layers import *
import torch_geometric.nn as gnn
from layers import *
from block import *

cuda = torch.device('cuda:0')
ddr_files_dir = 'DDR_files/'  # DDR files directory

# 10242 point model
class muilt_view_10242_ori(nn.Module):
    '''
    The architecture of NeuroExplainer (Neighborhood index of the vertex and the index of the vertex for template two)
    '''
    def __init__(self, in_ch, out_ch):
        '''
        Initialize the Spherical transformer.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        
        '''
        super(muilt_view_10242_ori, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42,neigh_orders_12 = Get_neighs_order_ori()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index()
        chs = [8, 16, 32, 64, 128, 256, 512]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        self.out_ch = out_ch
        self.down3 = down_block(conv_layer, in_ch, chs[2], neigh_orders_10242, None, True)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders_2562, neigh_orders_10242)
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders_642, neigh_orders_2562)
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders_162, neigh_orders_642)
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders_42, neigh_orders_162)
        self.point = point
        self.classify_42 = classify_block(chs[6], 2 * point[6], out_ch)  
        self.classify_162 = classify_block(chs[5], 2 * point[5], out_ch)
        self.SelfAttention = SelfAttention(in_dim=chs[5],head=6) 
        self.classify_642 = classify_block(chs[4], 2 * point[4], out_ch)
        self.classify_2562 = classify_block(chs[3], 2 * point[3], out_ch)
        self.classify_10242 = classify_block(chs[2], 2 * point[2], out_ch)
        self.up6 = up_block(conv_layer, chs[6], chs[5], neigh_orders_162, upconv_top_index_162, upconv_down_index_162)
        self.up5 = up_block(conv_layer, chs[5], chs[4], neigh_orders_642, upconv_top_index_642, upconv_down_index_642)
        self.up4 = up_block(conv_layer, chs[4], chs[3], neigh_orders_2562, upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders_10242, upconv_top_index_10242, upconv_down_index_10242)
        self.classify_end = nn.Linear(chs[2], 2) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
       
        x_class = []
        # encode block
        l_x5 = self.down3(x1)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        l_x2 = self.down6(l_x3)
        r_x5 = self.down3(x2)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        r_x2 = self.down6(r_x3)
        x2 = torch.cat((l_x2, r_x2), 2)
        x2 = x2.unsqueeze(2)
        x2 = self.SelfAttention(x2)
        x2 = x2.squeeze(2)
        # decode block and classify block
        x_class_162, x_new_162, x_new_minus_162, x_attention_162= self.classify_162(x2)
        x_class.append(x_class_162)
        l_x_162 = x_new_162[:, :, :self.point[5]]
        r_x_162 = x_new_162[:, :, self.point[5]:]
        l_x_642 = self.up5(l_x_162, l_x3)
        r_x_642 = self.up5(r_x_162, r_x3)
        x3 = torch.cat((l_x_642, r_x_642), 2)
        x_class_642, x_new_642, x_new_minus_642, x_attention_642 = self.classify_642(x3)
        x_class.append(x_class_642)
        l_x_642 = x_new_642[:, :, :self.point[4]]
        r_x_642 = x_new_642[:, :, self.point[4]:]
        l_x_2562 = self.up4(l_x_642, l_x4)
        r_x_2562 = self.up4(r_x_642, r_x4)
        x2 = torch.cat((l_x_2562, r_x_2562), 2)
        x_class_2562, x_new_2562, x_new_minus_2562, x_attention_2562= self.classify_2562(x2)
        x_class.append(x_class_2562)
        l_x_2562 = x_new_2562[:, :, :self.point[3]]
        r_x_2562 = x_new_2562[:, :, self.point[3]:]
        l_x_10242 = self.up3(l_x_2562, l_x5)
        r_x_10242 = self.up3(r_x_2562, r_x5)
        x1 = torch.cat((l_x_10242, r_x_10242), 2)
        # left and right hemisphere attention map
        x_class_10242, x_new_10242, x_new_minus_10242, x_attention_10242 = self.classify_10242(x1)
        x_class.append(x_class_10242)
        
        x_end = torch.mean(x_new_10242, 2)
        x_end = self.classify_end(x_end)
        x_end = self.softmax(x_end)
        x_class.append(x_end)
        # each layer classify result
        x_class = torch.stack(x_class, dim=2) 
        x_class = torch.mean(x_class, dim=2)  

        return x_class, x_new_162, x_new_minus_162, x_attention_162, x_new_10242, x_new_minus_10242, x_attention_10242

class muilt_view_10242(nn.Module):
    '''
    The architecture of NeuroExplainer (Neighborhood index of the vertex and the index of the vertex for template one)
    '''
    def __init__(self, in_ch, out_ch):
        super(muilt_view_10242, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42,neigh_orders_12 = Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index()
        chs = [8, 16, 32, 64, 128, 256, 512]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        self.out_ch = out_ch
        self.down3 = down_block(conv_layer, in_ch, chs[2], neigh_orders_10242, None, True)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders_2562, neigh_orders_10242)
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders_642, neigh_orders_2562)
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders_162, neigh_orders_642)
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders_42, neigh_orders_162)
       
        self.point = point
        self.classify_42 = classify_block(chs[6], 2 * point[6], out_ch)  #*2是左右两个半球
        self.classify_162 = classify_block(chs[5], 2 * point[5], out_ch)
        self.SelfAttention = SelfAttention(in_dim=chs[5],head=6) #注意力机制
        self.classify_642 = classify_block(chs[4], 2 * point[4], out_ch)
        self.classify_2562 = classify_block(chs[3], 2 * point[3], out_ch)
        self.classify_10242 = classify_block(chs[2], 2 * point[2], out_ch)
        self.up6 = up_block(conv_layer, chs[6], chs[5], neigh_orders_162, upconv_top_index_162, upconv_down_index_162)
        self.up5 = up_block(conv_layer, chs[5], chs[4], neigh_orders_642, upconv_top_index_642, upconv_down_index_642)
        self.up4 = up_block(conv_layer, chs[4], chs[3], neigh_orders_2562, upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders_10242, upconv_top_index_10242, upconv_down_index_10242)
        self.classify_end = nn.Linear(chs[2], 2)  #C_out
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        
        x_class = []
        # encode block
        l_x5 = self.down3(x1)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        l_x2 = self.down6(l_x3)
        r_x5 = self.down3(x2)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        r_x2 = self.down6(r_x3)
        
        x2 = torch.cat((l_x2, r_x2), 2)
        x2 = x2.unsqueeze(2)
        
        x2 = self.SelfAttention(x2)
        x2 = x2.squeeze(2)
        # decode block and classify block
        x_class_162, x_new_162, x_new_minus_162, x_attention_162= self.classify_162(x2)
        x_class.append(x_class_162)
        l_x_162 = x_new_162[:, :, :self.point[5]]
        r_x_162 = x_new_162[:, :, self.point[5]:]
        l_x_642 = self.up5(l_x_162, l_x3)
        r_x_642 = self.up5(r_x_162, r_x3)
        x3 = torch.cat((l_x_642, r_x_642), 2)
        x_class_642, x_new_642, x_new_minus_642, x_attention_642 = self.classify_642(x3)
        x_class.append(x_class_642)
        l_x_642 = x_new_642[:, :, :self.point[4]]
        r_x_642 = x_new_642[:, :, self.point[4]:]
        l_x_2562 = self.up4(l_x_642, l_x4)
        r_x_2562 = self.up4(r_x_642, r_x4)
        x2 = torch.cat((l_x_2562, r_x_2562), 2)
        x_class_2562, x_new_2562, x_new_minus_2562, x_attention_2562= self.classify_2562(x2)
        x_class.append(x_class_2562)
        l_x_2562 = x_new_2562[:, :, :self.point[3]]
        r_x_2562 = x_new_2562[:, :, self.point[3]:]
        l_x_10242 = self.up3(l_x_2562, l_x5)
        r_x_10242 = self.up3(r_x_2562, r_x5)
        x1 = torch.cat((l_x_10242, r_x_10242), 2)
        # left and right hemisphere attention map
        x_class_10242, x_new_10242, x_new_minus_10242, x_attention_10242 = self.classify_10242(x1)
        x_class.append(x_class_10242)
        
        x_end = torch.mean(x_new_10242, 2)
        x_end = self.classify_end(x_end)
        x_end = self.softmax(x_end)
        x_class.append(x_end)
        # each layer classify result
        x_class = torch.stack(x_class, dim=2)  
        x_class = torch.mean(x_class, dim=2)  

        return x_class, x_new_162, x_new_minus_162, x_attention_162, x_new_10242, x_new_minus_10242, x_attention_10242

# 40962 point model
class muilt_view_40962_ori(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(muilt_view_40962_ori, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42,neigh_orders_12 = Get_neighs_order_ori()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index_ori()

        chs = [16, 32, 64, 128, 256, 512, 1024]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        self.out_ch = out_ch
        self.down2 = down_block(conv_layer, in_ch, chs[1], neigh_orders_40962, None, True)
        self.down3 = down_block(conv_layer, chs[1], chs[2], neigh_orders_10242, neigh_orders_40962)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders_2562, neigh_orders_10242)
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders_642, neigh_orders_2562)
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders_162, neigh_orders_642)
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders_42, neigh_orders_162)
        
        self.point = point
        self.classify_42 = classify_block(chs[6], 2 * point[6], out_ch)
        self.classify_162 = classify_block(chs[5], 2 * point[5], out_ch)
        self.SelfAttention = SelfAttention(in_dim=chs[4], head=6)
        self.classify_642 = classify_block(chs[4], 2 * point[4], out_ch)
        self.classify_2562 = classify_block(chs[3], 2 * point[3], out_ch)
        self.classify_10242 = classify_block(chs[2], 2 * point[2], out_ch)
        self.classify_40962 = classify_block(chs[1], 2 * point[1], out_ch)
        self.up6 = up_block(conv_layer, chs[6], chs[5], neigh_orders_162, upconv_top_index_162, upconv_down_index_162)
        self.up5 = up_block(conv_layer, chs[5], chs[4], neigh_orders_642, upconv_top_index_642, upconv_down_index_642)
        self.up4 = up_block(conv_layer, chs[4], chs[3], neigh_orders_2562, upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders_10242, upconv_top_index_10242, upconv_down_index_10242)
        self.up2 = up_block(conv_layer, chs[2], chs[1], neigh_orders_40962, upconv_top_index_40962, upconv_down_index_40962)
        self.classify_end = nn.Linear(chs[1], 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # x1: left hemisphere, x2: right hemisphere  mini_batch * in_channel * point_num, x1 and x2 are the same
        x_class = []
        l_x6 = self.down2(x1)
        l_x5 = self.down3(l_x6)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        r_x6 = self.down2(x2)
        r_x5 = self.down3(r_x6)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        
        x3 = torch.cat((l_x3, r_x3), 2)
        x3 = x3.unsqueeze(2)
        # fuse the left and right hemisphere
        x3 = self.SelfAttention(x3)
        x3 = x3.squeeze(2)
        x_class_642, x_new_642, x_new_minus_642, x_attention_642 = self.classify_642(x3)
        x_class.append(x_class_642)
        l_x4 = self.up4(x_new_642[:, :, :self.point[4]], l_x4)
        r_x4 = self.up4(x_new_642[:, :, self.point[4]:], r_x4)
        x4 = torch.cat((l_x4, r_x4), 2)
        x_class_2562, x_new_2562, x_new_minus_2562, x_attention_2562= self.classify_2562(x4)
        x_class.append(x_class_2562)
        l_x5 = self.up3(x_new_2562[:, :, :self.point[3]], l_x5)
        r_x5 = self.up3(x_new_2562[:, :, self.point[3]:], r_x5)
        x5 = torch.cat((l_x5, r_x5), 2)
        x_class_10242, x_new_10242, x_new_minus_10242, x_attention_10242 = self.classify_10242(x5)
        x_class.append(x_class_10242)
        l_x6 = self.up2(x_new_10242[:, :, :self.point[2]], l_x6)
        r_x6 = self.up2(x_new_10242[:, :, self.point[2]:], r_x6)
        x6 = torch.cat((l_x6, r_x6), 2)
        # left and right hemisphere attention map
        x_class_40962, x_new_40962, x_new_minus_40962, x_attention_40962 = self.classify_40962(x6)
        x_class.append(x_class_40962)
        x_end = torch.mean(x_new_40962, 2)
        x_end = self.classify_end(x_end)
        x_end = self.softmax(x_end)
        x_class.append(x_end)
        # each layer classify result
        x_class = torch.stack(x_class, dim=2)
        x_class = torch.mean(x_class, dim=2)

        return x_class, x_new_642, x_new_minus_642, x_attention_642, x_new_40962, x_new_minus_40962, x_attention_40962

class muilt_view_40962(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(muilt_view_40962, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42,neigh_orders_12 = Get_neighs_order_ori()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index()

        chs = [16, 32, 64, 128, 256, 512, 1024]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        self.out_ch = out_ch
        self.down2 = down_block(conv_layer, in_ch, chs[1], neigh_orders_40962, None, True)
        self.down3 = down_block(conv_layer, chs[1], chs[2], neigh_orders_10242, neigh_orders_40962)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders_2562, neigh_orders_10242)
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders_642, neigh_orders_2562)
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders_162, neigh_orders_642)
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders_42, neigh_orders_162)
       
        self.point = point
        self.classify_42 = classify_block(chs[6], 2 * point[6], out_ch)
        self.classify_162 = classify_block(chs[5], 2 * point[5], out_ch)
        self.SelfAttention = SelfAttention(in_dim=chs[4], head=6)
        self.classify_642 = classify_block(chs[4], 2 * point[4], out_ch)
        self.classify_2562 = classify_block(chs[3], 2 * point[3], out_ch)
        self.classify_10242 = classify_block(chs[2], 2 * point[2], out_ch)
        self.classify_40962 = classify_block(chs[1], 2 * point[1], out_ch)
        self.up6 = up_block(conv_layer, chs[6], chs[5], neigh_orders_162, upconv_top_index_162, upconv_down_index_162)
        self.up5 = up_block(conv_layer, chs[5], chs[4], neigh_orders_642, upconv_top_index_642, upconv_down_index_642)
        self.up4 = up_block(conv_layer, chs[4], chs[3], neigh_orders_2562, upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders_10242, upconv_top_index_10242, upconv_down_index_10242)
        self.up2 = up_block(conv_layer, chs[2], chs[1], neigh_orders_40962, upconv_top_index_40962, upconv_down_index_40962)
        self.classify_end = nn.Linear(chs[1], 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # x1: left hemisphere, x2: right hemisphere  mini_batch * in_channel * point_num, x1 and x2 are the same
        x_class = []
        l_x6 = self.down2(x1)
        l_x5 = self.down3(l_x6)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        r_x6 = self.down2(x2)
        r_x5 = self.down3(r_x6)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        
        x3 = torch.cat((l_x3, r_x3), 2)
        x3 = x3.unsqueeze(2)
        # fuse the left and right hemisphere
        x3 = self.SelfAttention(x3)
        x3 = x3.squeeze(2)
        x_class_642, x_new_642, x_new_minus_642, x_attention_642 = self.classify_642(x3)
        x_class.append(x_class_642)
        l_x4 = self.up4(x_new_642[:, :, :self.point[4]], l_x4)
        r_x4 = self.up4(x_new_642[:, :, self.point[4]:], r_x4)
        x4 = torch.cat((l_x4, r_x4), 2)
        x_class_2562, x_new_2562, x_new_minus_2562, x_attention_2562= self.classify_2562(x4)
        x_class.append(x_class_2562)
        l_x5 = self.up3(x_new_2562[:, :, :self.point[3]], l_x5)
        r_x5 = self.up3(x_new_2562[:, :, self.point[3]:], r_x5)
        x5 = torch.cat((l_x5, r_x5), 2)
        x_class_10242, x_new_10242, x_new_minus_10242, x_attention_10242 = self.classify_10242(x5)
        x_class.append(x_class_10242)
        l_x6 = self.up2(x_new_10242[:, :, :self.point[2]], l_x6)
        r_x6 = self.up2(x_new_10242[:, :, self.point[2]:], r_x6)
        x6 = torch.cat((l_x6, r_x6), 2)
        # left and right hemisphere attention map
        x_class_40962, x_new_40962, x_new_minus_40962, x_attention_40962 = self.classify_40962(x6)
        x_class.append(x_class_40962)
        x_end = torch.mean(x_new_40962, 2)
        x_end = self.classify_end(x_end)
        x_end = self.softmax(x_end)
        x_class.append(x_end)
        # each layer classify result
        x_class = torch.stack(x_class, dim=2)
        x_class = torch.mean(x_class, dim=2)

        return x_class, x_new_642, x_new_minus_642, x_attention_642, x_new_40962, x_new_minus_40962, x_attention_40962

# 10242 point model generate negative result
class muilt_view_negative_10242(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(muilt_view_negative_10242, self).__init__()
        # size = 84*128
        neigh_orders = Get_neighs_order() 
        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42 = Get_upconv_index()

        chs = [8, 16, 32, 64, 128, 256, 512]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        # self.size = size
        self.out_ch = out_ch
        self.down3 = down_block(conv_layer, in_ch, chs[2], neigh_orders[2], None, True)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders[6], neigh_orders[5])
        
        # self.encode = Encode_10242(in_ch)
        self.point = point
        self.classify_42 = classify_block(chs[6], 2 * point[6], out_ch)
        self.classify_162 = classify_block(chs[5], 2 * point[5], out_ch)
        self.SelfAttention = SelfAttention(chs[5])  
        self.classify_642 = classify_block(chs[4], 2 * point[4], out_ch)
        self.classify_2562 = classify_block(chs[3], 2 * point[3], out_ch)
        self.classify_10242 = classify_negative_block(chs[2], 2 * point[2], out_ch)
        
        self.up6 = up_block(conv_layer, chs[6], chs[5], neigh_orders[5], upconv_top_index_162, upconv_down_index_162)
        self.up5 = up_block(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_642, upconv_down_index_642)
        self.up4 = up_block(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_10242, upconv_down_index_10242)
        self.outc = down_block(conv_layer, chs[2], in_ch, neigh_orders[2], None, True)
        self.classify_end = nn.Linear(chs[2], 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x_class = []
        l_x5 = self.down3(x1)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        l_x2 = self.down6(l_x3)
        r_x5 = self.down3(x2)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        r_x2 = self.down6(r_x3)

        x2 = torch.cat((l_x2, r_x2), 2)
        x2 = x2.unsqueeze(2)
        x2 = self.SelfAttention(x2)
        x2 = x2.squeeze(2)
        x_class_162, x_new_162, x_new_minus_162, x_attention_162= self.classify_162(x2)
        x_class.append(x_class_162)
        l_x_162 = x_new_162[:, :, :self.point[5]]
        r_x_162 = x_new_162[:, :, self.point[5]:]
        l_x_642 = self.up5(l_x_162, l_x3)
        r_x_642 = self.up5(r_x_162, r_x3)
        x3 = torch.cat((l_x_642, r_x_642), 2)
        x_class_642, x_new_642, x_new_minus_642, x_attention_642 = self.classify_642(x3)
        x_class.append(x_class_642)
        l_x_642 = x_new_642[:, :, :self.point[4]]
        r_x_642 = x_new_642[:, :, self.point[4]:]
        l_x_2562 = self.up4(l_x_642, l_x4)
        r_x_2562 = self.up4(r_x_642, r_x4)
        x2 = torch.cat((l_x_2562, r_x_2562), 2)
        x_class_2562, x_new_2562, x_new_minus_2562, x_attention_2562= self.classify_2562(x2)
        x_class.append(x_class_2562)
        l_x_2562 = x_new_2562[:, :, :self.point[3]]
        r_x_2562 = x_new_2562[:, :, self.point[3]:]
        l_x_10242 = self.up3(l_x_2562, l_x5)
        r_x_10242 = self.up3(r_x_2562, r_x5)
        x1 = torch.cat((l_x_10242, r_x_10242), 2)
        x_class_10242, x_new_10242, x_new_minus_10242, x_attention_10242 = self.classify_10242(x1)
        x_class.append(x_class_10242)
        x_end = torch.mean(x_new_10242, 2)  
        x_end = self.classify_end(x_end)  
        x_end = self.softmax(x_end)
        x_class.append(x_end)
        
        l_x_10242 = x_new_10242[:, :, :self.point[2]]
        r_x_10242 = x_new_10242[:, :, self.point[2]:]
        l_x = self.outc(l_x_10242)  
        r_x = self.outc(r_x_10242)
        x_class = x_end  

        return x_class, x_new_162, x_new_minus_162, l_x, r_x, x_attention_162, x_new_10242, x_new_minus_10242, x_attention_10242

# 10242 point model only encode
class muilt_view_encode_10242(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(muilt_view_encode_10242, self).__init__()
        neigh_orders = Get_neighs_order()
    
        chs = [8, 16, 32, 64, 128, 256, 512]

        point = [164842, 40962, 10242, 2562, 642, 162, 42]

        conv_layer = onering_conv_layer_batch

        self.out_ch = out_ch
        self.down3 = down_block(conv_layer, in_ch, chs[2], neigh_orders[2], None, True)
        self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
        self.down6 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])
        self.down7 = down_block(conv_layer, chs[5], chs[6], neigh_orders[6], neigh_orders[5])
        
        self.attention = nn.Linear(chs[5], out_ch) 
        self.SelfAttention = SelfAttention(chs[5])
        self.point = point
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x_class = []
        l_x5 = self.down3(x1)
        l_x4 = self.down4(l_x5)
        l_x3 = self.down5(l_x4)
        l_x2 = self.down6(l_x3)
        r_x5 = self.down3(x2)
        r_x4 = self.down4(r_x5)
        r_x3 = self.down5(r_x4)
        r_x2 = self.down6(r_x3)
        x2 = torch.cat((l_x2, r_x2), 2)
        x2 = x2.unsqueeze(2)
        x2 = self.SelfAttention(x2)
        x2 = x2.squeeze(2)
        x2 = x2.permute(0, 2, 1)
        batch, point, channel = x2.size()
        x_gap = torch.mean(x2, 1) 
        x_class = self.attention(x_gap)   
        x_class = self.softmax(x_class)

        return x_class, x2