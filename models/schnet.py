import os
import os.path as osp
import warnings
from math import pi as PI

import ase
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

try:
    import schnetpack as spk
except ImportError:
    spk = None

#连续滤波卷积 捕捉原子间的滤波关系
class SchNet(torch.nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=4, num_gaussians=50, cutoff=10.0,  #cutoff距离截断阈值 用于忽略远距离无作用的原子
                 readout='mean', dipole=False, mean=None, std=None, atomref=None):
        super(SchNet, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.readout = 'add' if dipole else readout
        self.num_interactions = num_interactions
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        # self.readout = readout
        self.cutoff = cutoff
        self.dipole = dipole
        self.scale = None
        self.mean = mean
        self.std = std
        self.edge_index = None
        self.edge_attr = None

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        # TODO: double-check hidden size
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        # self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None, precomputed_edge_weight=None, precomputed_edge_index=None):
        # assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        # h = self.embedding(z)
        h = z
        # print('pos', pos.shape)

        # 如果提供了预计算的edge_index和edge_weight，且edge_index不为空，则使用它们
        # 否则重新计算
        if precomputed_edge_index is not None and precomputed_edge_index.numel() > 0:
            edge_index = precomputed_edge_index
            # 如果提供了预计算的edge_weight，使用它；否则计算
            if precomputed_edge_weight is not None and precomputed_edge_weight.numel() > 0:
                edge_weight = precomputed_edge_weight
            else:
                row, col = edge_index
                edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        else:
            # 默认行为：使用radius_graph重新构建图
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1) #计算原子对间的欧式距离
        
        self.edge_index = edge_index
        edge_attr = self.distance_expansion(edge_weight) #将离散距离编码转为连续高斯向量？
        self.edge_attr = edge_attr

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        #输出层 非线性变换
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        out = h

        out[~torch.isfinite(out)] = 0.0
        return out

    def get_edge_index(self):
        return self.edge_index

    def get_edge_weight(self):
        return self.edge_attr

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')

#交互块 原子特征迭代更新
class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        #基于距离生成动态滤波权重
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        #连续滤波卷积
        self.conv = CFConv(hidden_channels, hidden_channels, 
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

#连续滤波卷积
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn  #from InteractionBlock
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)  #距离越近权重越大
        W = self.nn(edge_attr) * C.view(-1, 1)  #动态生成滤波权重

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W

#高斯平滑
class GaussianSmearing(torch.nn.Module):

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

#偏移软激活函数，引入非线性且避免梯度消失
class ShiftedSoftplus(torch.nn.Module):

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SchNet()
num_params = count_parameters(model)
print(f'The model has {num_params} parameters.')