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
import scipy.io as sio

cuda = torch.device('cuda:0')   
ddr_files_dir = 'DDR_files/'  

class up_layer_norm(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter.
    The features of the added points are linearly interpolated.

    Parameters
    __________
    in_feats : int
            # of input channels in this layer.
    out_feats : int
            # of output channels in this layer.

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        [Batch, out_feats(feature), (Nx4)-6(vertex)], tensor

    """
    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(up_layer_norm, self).__init__()  
    
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index 
        self.upconv_down_index = upconv_down_index 

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, raw_nodes, channel = x.size()
        new_nodes = int(raw_nodes * 4 - 6)
        x = torch.cat([x, x, x, x, x, x, x], dim=2)
        x = x.view(batch, raw_nodes * 7, self.out_feats) 
        x1 = x[:, self.upconv_top_index, :]  
        assert (x1.size() == torch.Size([batch, raw_nodes, self.out_feats]))  
        x2 = x[:, self.upconv_down_index, :].view(batch, -1, self.out_feats, 2)  
        x = torch.cat((x1, torch.mean(x2, 3)), 1)  
        assert (x.size() == torch.Size([batch, new_nodes, self.out_feats]))
        x = x.permute(0, 2, 1)
        return x
    

class onering_conv_layer_batch(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using 1-ring filter.

    Parameters
    __________
    in_feats : int
            # of input channels in this layer.
    out_feats : int
            # of output channels in this layer.
    neigh_orders :int  N x 7   numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        [Batch, out_feats(feature), N(vertex)], tensor

    """

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None):
        super(onering_conv_layer_batch, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders 

        self.weight = nn.Linear(7 * in_feats, out_feats)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, point, channel = x.size()
        x = x[:, self.neigh_orders, :]#B*N*7*out_feats
        mat = x.view(batch, point, 7 * self.in_feats)#变宽
        out_features = self.weight(mat)
        out_features = out_features.permute(0, 2, 1)
        return out_features

class tworing_conv_layer_batch(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using 2-ring filter.

    Parameters
    __________
    in_feats : int
            # of input channels in this layer.
    out_feats : int
            # of output channels in this layer.
    neigh_orders :int  N x 19   numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        [Batch, out_feats(feature), N(vertex)], tensor

    """

    def __init__(self, in_feats, out_feats, neigh_orders):
        super(tworing_conv_layer_batch, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(19 * in_feats, out_feats)  
        
    def forward(self, x):
       
        x = x.permute(0, 2, 1)
        batch, point, channel = x.size()
        x = x[:, self.neigh_orders, :]
        mat = x.view(batch, point, 19 * self.in_feats)

        out_features = self.weight(mat)
        out_features = out_features.permute(0, 2, 1)
        return out_features

class pool_layer_batch(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere using 1-ring filter.

    Parameters
    __________
    neigh_orders :int    numpy.ndarray
            # Neighborhood index of the vertex and the index of the vertex

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        [Batch, out_feats(feature), (N+6)/4(vertex)], tensor

    """

    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer_batch, self).__init__()

        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type

    def forward(self, x):

        batch, feat_num, num_nodes = x.size()
        num_nodes = int((num_nodes + 6) / 4)
        x = x[:, :, self.neigh_orders[0:num_nodes * 7]].view(batch, feat_num, num_nodes,  7)
        if self.pooling_type == "mean":
            x = torch.mean(x, 3)
        if self.pooling_type == "max":
            x = torch.max(x, 3)
            assert (x[0].size() == torch.Size([num_nodes, feat_num]))
            return x[0], x[1]

        assert (x.size() == torch.Size([batch, feat_num, num_nodes]))

        return x

class upconv_layer_batch(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter.
    The features of the added points are linearly interpolated.

    Parameters
    __________
    in_feats : int
            # of input channels in this layer.
    out_feats : int
            # of output channels in this layer.
    upconv_top_index : int
            # Index of the vertex before upsampling
    upconv_down_index : int
            # Parent node index of the point added after upsampling

    Notes
    —————
    Input shape :
        [Batch, in_feats(feature), N(vertex)], tensor
    Output shape:
        [Batch, out_feats(feature), (4N-6)/(vertex)], tensor

    """

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer_batch, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(in_feats, 7 * out_feats)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch, raw_nodes, channel = x.size()
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x)
        x = x.view(batch, raw_nodes * 7, self.out_feats)
        x1 = x[:, self.upconv_top_index, :]
        assert (x1.size() == torch.Size([batch, raw_nodes, self.out_feats]))
        x2 = x[:, self.upconv_down_index, :].view(batch, -1, self.out_feats, 2)
        x = torch.cat((x1, torch.mean(x2, 3)), 1)
        assert (x.size() == torch.Size([batch, new_nodes, self.out_feats]))
        x = x.permute(0, 2, 1)
        return x
    
def Get_neighs_order():
    neigh_orders_40962 = get_neighs_order('neigh/neigh_order_40962.mat')
    neigh_orders_10242 = get_neighs_order('neigh/neigh_order_10242.mat')
    neigh_orders_2562 = get_neighs_order('neigh/neigh_order_2562.mat')
    neigh_orders_642 = get_neighs_order('neigh/neigh_order_642.mat')
    neigh_orders_162 = get_neighs_order('neigh/neigh_order_162.mat')
    neigh_orders_42 = get_neighs_order('neigh/neigh_order_42.mat')
    neigh_orders_12 = get_neighs_order('neigh/neigh_order_12.mat')
    
    return neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12

def Get_neighs_order_ori():
    neigh_orders_40962 = get_neighs_order_ori('neigh_indices/adj_mat_order_40962.mat')
    neigh_orders_10242 = get_neighs_order_ori('neigh_indices/adj_mat_order_10242.mat')
    neigh_orders_2562 = get_neighs_order_ori('neigh_indices/adj_mat_order_2562.mat')
    neigh_orders_642 = get_neighs_order_ori('neigh_indices/adj_mat_order_642.mat')
    neigh_orders_162 = get_neighs_order_ori('neigh_indices/adj_mat_order_162.mat')
    neigh_orders_42 = get_neighs_order_ori('neigh_indices/adj_mat_order_42.mat')
    neigh_orders_12 = get_neighs_order_ori('neigh_indices/adj_mat_order_12.mat')
    
    return neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12

def get_neighs_order(order_path):
    '''
    Neighborhood index of one-ring for template one.

    Args:
        order_path: Path of the data

    Returns:
        neigh_orders: Neighborhood index of the vertex and the index of the vertex

    '''
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['neigh_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    return neigh_orders

def get_neighs_order_ori(order_path):
    '''
        Neighborhood index of one-ring for template two.

        Args:
            order_path: Path of the data

        Returns:
            neigh_orders: Neighborhood index of the vertex and the index of the vertex

    '''
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)  
    
    return neigh_orders

def Get_upconv_index():
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index('neigh/neigh_order_40962.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index('neigh/neigh_order_10242.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index('neigh/neigh_order_2562.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index('neigh/neigh_order_642.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index('neigh/neigh_order_162.mat')
    upconv_top_index_42, upconv_down_index_42 = get_upconv_index('neigh/neigh_order_42.mat')
    
    return upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42

def Get_upconv_index_ori():
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index_ori('neigh_indices/adj_mat_order_40962.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index_ori('neigh_indices/adj_mat_order_10242.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index_ori('neigh_indices/adj_mat_order_2562.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index_ori('neigh_indices/adj_mat_order_642.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index_ori('neigh_indices/adj_mat_order_162.mat')
    upconv_top_index_42, upconv_down_index_42 = get_upconv_index_ori('neigh_indices/adj_mat_order_42.mat')
    
    return upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42


def get_upconv_index(order_path):
    '''
    Get the index of the vertex before upsampling and parent node index of the point added after upsampling for template one.

    Args:
        order_path: order_path: Path of the data

    Returns:
        upconv_top_index : Index of the vertex before upsampling
        upconv_down_index : Parent node index of the point added after upsampling

    '''
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['neigh_order']
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index

def get_upconv_index_ori(order_path):
    '''
        Get the index of the vertex before upsampling and parent node index of the point added after upsampling for template two.

        Args:
            order_path: order_path: Path of the data

        Returns:
            upconv_top_index : Index of the vertex before upsampling
            upconv_down_index : Parent node index of the point added after upsampling

    '''
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index

def Get_2ring_neighs_order():
    neigh_orders_2ring_40962 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_40962.mat')
    neigh_orders_2ring_10242 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_10242.mat')
    neigh_orders_2ring_2562 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_2562.mat')
    neigh_orders_2ring_642 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_642.mat')
    neigh_orders_2ring_162 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_162.mat')
    neigh_orders_2ring_42 = get_2ring_neighs_order('neigh/adj_mat_order_2ring_42.mat')
    
    return neigh_orders_2ring_40962, neigh_orders_2ring_10242, neigh_orders_2ring_2562, neigh_orders_2ring_642, neigh_orders_2ring_162, neigh_orders_2ring_42

def get_2ring_neighs_order(order_path):
    '''
    Neighborhood information of two-ring for template two.

    Args:
        order_path: Path of the data

    Returns:
        neigh_orders: Neighborhood index of the vertex and the index of the vertex

    '''
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order_2ring']
    neigh_orders = np.zeros((len(adj_mat_order), 19))
    neigh_orders[:,0:18] = adj_mat_order-1
    neigh_orders[:,18] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders
