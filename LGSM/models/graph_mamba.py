import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GINConv, BatchNorm, global_add_pool, radius_graph
from torch_geometric.nn import MLP as PyGMLP
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from einops import rearrange, repeat, einsum
from typing import Union
from dataclasses import dataclass
import math
from torch_geometric.nn import GINEConv, MLP, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from models.schnet import SchNet, GaussianSmearing
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor
from torch_geometric.utils import degree, sort_edge_index, to_dense_batch, to_dense_adj
from models.gnn_model import GNN

class PosEmbedSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    #num_pos_feats位置编码的特征维度（值越大越精细）scale控制位置特征编码范围的缩放参数 temperature用于控制位置编码中不同频率分量的衰减速度
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: torch.tensor, shape (L, d)

        Returns:
            pos_x: Position embeddings of shape (L, num_pos_feats)
        """
        L = x.size(0)  # 输入序列长度
        mask = torch.ones(L, device=x.device)
        # 创建 mask，形状为 (L,)
        x_embed = mask.cumsum(0, dtype=torch.float32)  # (L,)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[-1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t  # (L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1).flatten(1)

        return pos_x


class CrossAttentionFusion(nn.Module):
    """
    使用交叉注意力将官能团信息融合到原子序列中。
    Query: 原子序列
    Key/Value: 官能团序列
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 通常不在这里加额外的投影层，因为输入已经投影过了
        # 但如果需要，可以添加 self.query_proj, self.key_proj, self.value_proj

    def forward(self, atoms, func_groups, func_group_mask=None):
        """
        atoms: [b, n, d_model] - 主干原子序列
        func_groups: [b, fn, d_model] - 官能团序列
        func_group_mask: [b, fn] (optional) - 用于忽略填充的官能团
        """
        # 计算注意力分数 Q @ K^T
        attn_scores = torch.matmul(atoms, func_groups.transpose(-1, -2))
        attn_scores = attn_scores / (self.d_model ** 0.5)

        # (可选但推荐) 应用 mask，防止注意力被分配到 [PAD] token 上
        if func_group_mask is not None:
            # Mask shape [b, fn] -> [b, 1, fn] for broadcasting
            attn_scores = attn_scores.masked_fill(func_group_mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # -> [b, n, fn]

        # 用权重聚合 Value
        # (attn_weights @ V) -> ([b, n, fn] @ [b, fn, d_model]) -> [b, n, d_model]
        func_group_context = torch.matmul(attn_weights, func_groups)

        # 将上下文与原始原子信息结合
        # 这里使用简单的相加，也可以用层归一化或门控机制
        fused_atoms = atoms + func_group_context

        return fused_atoms

class FiLMGenerator(nn.Module):
    """
    从全局上下文中生成 FiLM 的 gamma 和 beta 参数。
    """

    def __init__(self, d_model):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )

    def forward(self, context_vector):
        """
        context_vector: [b, d_model] - 全局上下文
        """
        gamma_beta = self.generator(context_vector)  # -> [b, 2 * d_model]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # each -> [b, d_model]
        return gamma, beta


class ResidualBlock(nn.Module):
    def __init__(self,
                 gnn,
                 d_model: int = 128,
                 obd: bool = False,
                 obf: bool = False,
                 cutoff: float = 10.0,
                 ):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.gnn = gnn
        self.cutoff = cutoff
        self.lin = nn.Linear(50, 1)
        self.distance_expansion = GaussianSmearing(0.0, 10, 50)

        self.degree_emb = PosEmbedSine(num_pos_feats=d_model, normalize=True)
        self.frag_emb = PosEmbedSine(num_pos_feats=d_model, normalize=True)
        self.order_by_degree = obd
        self.order_by_frag = obf

        # self.mixer = MambaBlock(d_model)
        self.mixer = MambaBlockBatched(d_model)

    def forward(self, x, pos, batch, gpt_emb=None, fg_emb=None, precomputed_radius_edge_index=None, precomputed_radius_edge_weight=None):
        x1 = self.norm(x)  # [n, d]
        
        # 优先使用预计算的radius_edge_index和radius_edge_weight（如果提供且节点顺序未改变）
        # 如果提供了预计算值且节点顺序未改变（第一层未排序或后续层），则使用
        if (precomputed_radius_edge_index is not None and 
            precomputed_radius_edge_index.numel() > 0 and
            not (self.order_by_degree or self.order_by_frag)):
            # 使用预计算的值，但需要检查节点数量是否匹配
            if precomputed_radius_edge_index.max() < x.shape[0]:
                pre_edge_index = precomputed_radius_edge_index
                pre_edge_weight = precomputed_radius_edge_weight
            else:
                # 节点数量不匹配，重新计算
                pre_edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
                if pre_edge_index.numel() == 0:
                    raise ValueError("pre_edge_index is empty. Please check the input data.")
                row_pre, col_pre = pre_edge_index
                pre_edge_weight = (pos[row_pre] - pos[col_pre]).norm(dim=-1)
        else:
            # 没有预计算值或节点顺序已改变，重新计算
            pre_edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            if pre_edge_index.numel() == 0:
                raise ValueError("pre_edge_index is empty. Please check the input data.")
            row_pre, col_pre = pre_edge_index
            pre_edge_weight = (pos[row_pre] - pos[col_pre]).norm(dim=-1)

        # 使用计算的或预计算的半径图与距离，避免 SchNet 内部重复构图与计算距离
        x2 = self.gnn(
            x1,
            pos,
            batch,
            precomputed_edge_weight=pre_edge_weight,
            precomputed_edge_index=pre_edge_index,
        )

        # if self.order_by_frag:
        #     order_frag = torch.stack([batch, map], 1).T
        #     sort_frag, x2 = sort_edge_index(order_frag, edge_attr=x2)
        #     _, map = sort_edge_index(order_frag, edge_attr=map)
        #     _, pos = sort_edge_index(order_frag, edge_attr=pos)
        #     _, x1 = sort_edge_index(order_frag, edge_attr=x1)
        #
        #     # print('x20', x2.shape)
        #     x2 = x2 + self.frag_emb(x2)

        # sort based on the degree of nodes, batch
        if self.order_by_degree:
            deg = degree(pre_edge_index[0], x2.shape[0]).to(torch.long)
            order_deg = torch.stack([batch, deg], 1).T
            sort_deg, x2 = sort_edge_index(order_deg, edge_attr=x2)
            # _, map = sort_edge_index(order_deg, edge_attr=map)
            _, pos = sort_edge_index(order_deg, edge_attr=pos)
            _, x1 = sort_edge_index(order_deg, edge_attr=x1)

            x2 = x2 + self.degree_emb(x2)

        # 注意：若上方发生了排序，pre_edge_index 已不再与当前 pos 对齐，因此此处仍需基于当前 pos 重新构建
        e_t = radius_graph(pos, r=self.cutoff, batch=batch)

        # 将节点特征 x2 转换为填充后的批次张量 [b, max_nodes, d]
        x_batch, node_mask = to_dense_batch(x2, batch)
        b, max_nodes, d = x_batch.shape

        #计算边的属性并构建批次化的邻接矩阵 [b, max_nodes, max_nodes]
        if e_t.numel() == 0:
            # 如果整个批次都没有任何边，创建一个单位矩阵作为邻接矩阵
            # 注意：原始代码的 ones(10, 10) 可能是个bug或硬编码，这里使用更通用的单位矩阵
            dis_dense_batch = torch.eye(max_nodes, device=x2.device).unsqueeze(0).repeat(b, 1, 1)
        else:
            row, col = e_t
            # 注意：pos 已经是排序后的，所以这里的索引是正确的
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            d_t = self.distance_expansion(edge_weight)
            d_t = self.lin(d_t).squeeze(1)

            dis_dense_batch = to_dense_adj(edge_index=e_t, batch=batch, edge_attr=d_t, max_num_nodes=max_nodes)

        # 将批次化数据送入批处理版的 MambaBlock
        # self.mixer 接收 [b, l, d] 和 [b, l, l] 形状的张量
        output_batch = self.mixer(x_batch, dis_dense_batch, gpt_emb, fg_emb)

        # 将填充后的批次张量转换回 PyG 的大图格式 [total_nodes, d]
        x3 = output_batch[node_mask]
        output = x3 + x1
        return output, pos


class Graph_Mamba(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 n_layer: int = 4,
                 sch_layer: int = 4,
                 dim_in: int = 2,
                 cutoff: float = 5.0,
                 ):
        super().__init__()

        # self.gin = GNN(num_layer=4, emb_dim=d_model)
        self.gnn = SchNet(hidden_channels=d_model,
                          num_filters=d_model,
                          num_interactions=sch_layer,
                          num_gaussians=50,
                          cutoff=cutoff,
                          readout='mean',
                          )
        self.encode = nn.Linear(dim_in, d_model)
        if n_layer == 1:
            pass
        else:
            self.encoder_layers = nn.ModuleList()
            self.encoder_layers.append(ResidualBlock(self.gnn, d_model, obd=True, obf=True, cutoff=cutoff))
            for _ in range(n_layer-1):
                self.encoder_layers.append(ResidualBlock(self.gnn, d_model, obd=False, obf=False, cutoff=cutoff))

        # readout
        self.encoder_norm = RMSNorm(d_model)
        self.decode = nn.Linear(d_model, d_model)

        self.frag_pred = nn.Linear(d_model, 3200)
        self.tree_pred = nn.Linear(d_model, 3000)
        self.pool = global_mean_pool

    def forward(self, data, gpt_emb=None, fg_emb=None):
        x, pos, edge_index, batch = data.x.float(), data.pos, data.edge_index, data.node_batch
        
        # 尝试使用预计算的radius_edge_index和radius_edge_weight
        # 如果数据中有这些属性，则使用；否则训练时会动态计算
        precomputed_radius_edge_index = getattr(data, 'radius_edge_index', None)
        precomputed_radius_edge_weight = getattr(data, 'radius_edge_weight', None)
        
        x = self.encode(x)  # [n, d]
        # x = self.gin(x, edge_index, data.edge_attr)
        for i, layer in enumerate(self.encoder_layers):
            # 第一层可能进行排序，排序后预计算的索引不再匹配
            # 所以只在第一层（如果未排序）或后续层使用预计算值
            use_precomputed = (i == 0 and not (layer.order_by_degree or layer.order_by_frag)) or (i > 0)
            if use_precomputed and precomputed_radius_edge_index is not None and precomputed_radius_edge_index.numel() > 0:
                x, pos = layer(x, pos, batch, gpt_emb, fg_emb,
                                    precomputed_radius_edge_index=precomputed_radius_edge_index,
                                    precomputed_radius_edge_weight=precomputed_radius_edge_weight)
            else:
                x, pos = layer(x, pos, batch, gpt_emb, fg_emb)
        x = self.encoder_norm(x)
        x = self.decode(x)

        x[~torch.isfinite(x)] = 0.0
        # frag_emb = scatter(x, map, dim=0, reduce="mean")
        # pred_frag = self.pool(self.frag_pred(x), batch)
        # pred_tree = self.pool(self.tree_pred(x), batch)
        # pred_tree[~torch.isfinite(pred_tree)] = 0.0
        # pred_frag[~torch.isfinite(pred_frag)] = 0.0

        return x


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 bias: bool = False,
                 conv_bias: bool = True,
                 d_conv: int = 4,
                 dt_rank: Union[int, str] = 'auto',
                 d_state: int = 2,
                 ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)
        self.d_model = d_model

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, dis_dense):
        x = x.unsqueeze(0)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_model, self.d_model], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x, dis_dense)

        y = y * F.silu(res)

        output = self.out_proj(y)

        output = output.squeeze(0)
        output[~torch.isfinite(output)] = 0.0

        return output

    def ssm(self, x, dis_dense):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D, dis_dense)

        return y

    def selective_scan(self, u, delta, A, B, C, D, dis_dense):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        temp_adj = dis_dense
        if temp_adj.size(0) >= d_in:
            temp_adj = temp_adj[:d_in, :d_in]
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)
        # print(temp_adj_padded.shape)
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj

        delta_p = torch.matmul(delta, temp_adj_padded)

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class MambaBlockBatched(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 bias: bool = False,
                 conv_bias: bool = True,
                 d_conv: int = 4,
                 dt_rank: Union[int, str] = 'auto',
                 d_state: int = 2,
                 ):
        super().__init__()

        self.attention_fusion = CrossAttentionFusion(d_model)
        self.film_generator = FiLMGenerator(d_model)

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)
        self.d_model = d_model

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)


    def forward(self, atoms, dis_dense, gpt_emb=None, fg_emb=None):
        x = atoms
        if fg_emb is not None:
            x = self.attention_fusion(atoms, fg_emb)

        if gpt_emb is not None:
            gamma, beta = self.film_generator(gpt_emb)
            x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)

        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_model, self.d_model], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x, dis_dense)
        y = y * F.silu(res)
        output = self.out_proj(y)
        output[~torch.isfinite(output)] = 0.0

        return output

    def ssm(self, x, dis_dense):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D, dis_dense)

        return y

    def selective_scan(self, u, delta, A, B, C, D, dis_dense):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # --- 对 dis_dense 的批次处理 ---
        # 假设 dis_dense 是一个形状为 [b, d_in, d_in] 的批次化邻接矩阵
        # 如果 dis_dense 的维度与 d_in 不匹配，可能需要进行批次化的裁剪或填充
        # 注意：这里的逻辑需要根据你实际的 dis_dense 格式进行调整
        temp_adj_padded = dis_dense  # 简化假设，传入的已经是正确的形状 [b, d_in, d_in]
        delta_p = torch.matmul(temp_adj_padded, delta)

        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # --- 循环本身已经是批次化的 ---
        # x 的形状是 [b, d_in, n]，它同时为批次中的每个样本维护一个隐藏状态
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            # 所有操作（*、+、einsum）都作用在批次维度 b 上，是并行的
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == '__main__':
    from torch_geometric.data import Data
    from datasets.loader_pretrain import MoleculePretrainDataset, mol_frag_collate
    from torch.utils.data import DataLoader

    # Instantiate the model
    model = Graph_Mamba(d_model=128, n_layer=4, dim_in=2).cuda()

    root = r''
    data_file_path = r''
    vocab_file_path = r'.\datasets\vocab.txt'

    dataset = MoleculePretrainDataset(root=root,
                                      smiles_column='smiles',
                                      data_file_path=data_file_path,
                                      vocab_file_path=vocab_file_path)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=mol_frag_collate)

    for data in loader:
        x, frag_emb, pred_frag, pred_tree = model(data.cuda())
        print(x.shape, frag_emb.shape, pred_frag.shape, pred_tree.shape)
        break