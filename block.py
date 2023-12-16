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
from sphericalunet.utils.utils import *
from sphericalunet.models.layers import *
import torch_geometric.nn as gnn
from layers import *

cuda = torch.device('cuda:0')
ddr_files_dir = 'DDR_files/'  

def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)  
    output = torch.div(input, norm) 
    return output

def gmm_conv(in_ch, out_ch, kernel_size=3):
    return gnn.GMMConv(in_ch, out_ch, dim=2, kernel_size=kernel_size)  

def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_() 


class SelfAttention(nn.Module):
    """
    Self attention Layer.
    Source paper: https://arxiv.org/abs/1805.08318
    Input:
        x : input feature maps( B X C X W X H)   batch*channel*width *height
    Returns :
        self attention feature maps

    """

    def __init__(self, in_dim, head = 6, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1) 
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.head = head

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, (self.head*C)//8, (width * height)//self.head)  
        g = self.g(x).view(m_batchsize, (self.head*C)//8, (width * height)//self.head)   
        h = self.h(x).view(m_batchsize, self.head*C, (width * height)//self.head)  

        attention = torch.bmm(f.permute(0, 2, 1), g)  
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  
        self_attetion = self_attetion.view(m_batchsize, C, width, height) 

        out = self.gamma * self_attetion + x
        return out


class down_block(nn.Module):
    """
    Downsampling block in model.
    mean pooling => (conv => BN => ReLU) * 2

    Parameters
    __________
    conv_layer : 
            # convolutional layer
    in_ch : int
            # of input channels in this layer.
    out_ch : int
            # of output channels in this layer.
    neigh_orders : int    numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex after pooling.
    pool_neigh_orders: int    numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex before pooling.
    first : bool   
            # pooling or not

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        first = Flase : [Batch, out_feats(feature), (N+6)/4(vertex)], tensor
        first = True : [Batch, out_feats(feature), N(vertex)], tensor

    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
        super(down_block, self).__init__()

        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True)
            )  
        else:
            self.block = nn.Sequential(
                pool_layer_batch(pool_neigh_orders),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5)
            )
    def forward(self, x):
        x = self.block(x)
        return x

class up_block(nn.Module):
    """
    Upsamping block in model.
    upconv => (conv => BN => ReLU) * 2

    Parameters
    __________
    conv_layer : 
            # convolutional layer
    in_ch : int
            # of input channels in this layer.
    out_ch : int
            # of output channels in this layer.
    neigh_orders : int    numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex.
    upconv_top_index : int
            # Index of the vertex before upsampling
    upconv_down_index : int
            # Parent node index of the point added after upsampling


    """

    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()

        self.up = upconv_layer_batch(in_ch, out_ch, upconv_top_index, upconv_down_index)

        self.double_conv = nn.Sequential(
            conv_layer(in_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            conv_layer(out_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1)
        x = self.double_conv(x)

        return x

class classify_block(nn.Module):
    '''
    The attention Mechanism in model.(quantile normalization for attention)
    Parameters
    __________
    channel : int
            # of input channels in this layer.
    height: int
            # length of data
    out_class : int
            # Number of categories.
    
    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        x_class: [Batch, out_class], tensor
        x_attention: [Batch, N(vertex)], tensor
    '''
    
    def __init__(self, channel, height, out_class=2):
        super(classify_block, self).__init__()
        self.out_ch = out_class
        self.channel = channel
        self.height = height
        size = height * channel
        self.size = size
        self.attention = nn.Linear(channel, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()  
        self.tanh = nn.Tanh()   
        self.bias = nn.Linear(2, 1)
        self.Norm = nn.LayerNorm(height)   
        self.softmax_attention = nn.Softmax(dim=1)
        self.batchnorm = nn.BatchNorm1d(height)  
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, point, channel = x.size()
        x_gap = torch.mean(x, 1) 
        x_class = self.attention(x_gap) 
        x_class = self.softmax(x_class)  
        x_attention = self.attention(x)  
        x_class_minus = 1 - x_class     
        x_attention = (x_attention * x_class.unsqueeze(1)).sum(dim=2)
        x_attention = self.batchnorm(x_attention)  
        x_attention_max = torch.quantile(x_attention, 0.99, dim=1)
        x_attention_min = torch.quantile(x_attention, 0.001, dim=1)
        x_attention = torch.clamp(x_attention, x_attention_min.unsqueeze(1), x_attention_max.unsqueeze(1))
        x_attention = (x_attention - x_attention_min.unsqueeze(1) + 1e-6) / (x_attention_max - x_attention_min + 1e-6).unsqueeze(1)
        x_attention_minus = 1 - x_attention
        x = x.permute(0, 2, 1)
        x_new = x * x_attention.unsqueeze(1)  
        x_new_minus = x * x_attention_minus.unsqueeze(1)

        return x_class, x_new, x_new_minus, x_attention


class classify_negative_block(nn.Module):
    '''
    The attention Mechanism in model.(max-min normalization for attention)
    Parameters
    __________
    channel : int
            # of input channels in this layer.
    height: int
            # length of data
    out_class : int
            # Number of categories.
    
    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        x_class: [Batch, out_class], tensor
        x_attention: [Batch, N(vertex)], tensor
    '''
    def __init__(self, channel, height, out_class=2):
        super(classify_negative_block, self).__init__()
        self.out_ch = out_class
        self.channel = channel
        self.height = height
        size = height * channel
        self.size = size
        self.attention = nn.Linear(channel, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bias = nn.Linear(2, 1)
        self.Norm = nn.LayerNorm(height)
        self.softmax_attention = nn.Softmax(dim=1)
        self.batchnorm = nn.BatchNorm1d(height)

    #x size：B * C * Point
    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, point, channel = x.size()
        x_gap = torch.mean(x, 1)
        x_class = self.attention(x_gap)
        x_class = self.softmax(x_class)
        x_attention = self.attention(x)
        x_class_minus = 1 - x_class
        x_attention = (x_attention * x_class.unsqueeze(1)).sum(dim=2)
        x_attention = self.batchnorm(x_attention)
        x_attention_min = torch.min(x_attention, dim=1)[0] #按列求最小，返回values和indices
        x_attention_max = torch.max(x_attention, dim=1)[0]
        x_attention = (x_attention - x_attention_min.unsqueeze(1) + 1e-6) / (x_attention_max - x_attention_min + 1e-6).unsqueeze(1)
        x_attention_minus = x_attention
        x_attention = 1 - x_attention
        x = x.permute(0, 2, 1)
        x_new = x * x_attention.unsqueeze(1)
        x_new_minus = x * x_attention_minus.unsqueeze(1)

        return x_class, x_new, x_new_minus, x_attention


