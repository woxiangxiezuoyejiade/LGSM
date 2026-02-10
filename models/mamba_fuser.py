import torch
import torch.nn as nn
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch.functional import F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from models.graph_mamba import Graph_Mamba, PosEmbedSine
from models.gnn_model import GINE, GNN
from models.mamba import Mamba
from models.loss_info_nce import InfoNCE
from torch_geometric.utils import add_self_loops


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(5, emb_dim) 
        self.edge_embedding2 = torch.nn.Embedding(3, emb_dim) 

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr != None:
            # add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        else:
            edge_embeddings = torch.zeros((edge_index[0].shape[1], x.shape[-1])).to(x.device)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GIN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0, atom=False):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom:
            self.x_embedding1 = torch.nn.Embedding(120, emb_dim) #原子序数嵌入
            self.x_embedding2 = torch.nn.Embedding(3, emb_dim) #原子电荷嵌入

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.embedding = nn.Embedding(908, emb_dim) #分子片段id嵌入

        self.atom = atom
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim))
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.atom:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        else:
            x = self.embedding(x[:, 0])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # print(h.shape)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        node_representation = h_list[-1]

        return node_representation




class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz).cuda()
        layers = [
            nn.Dropout(dropout).cuda(),
            nn.Linear(in_hsz, out_hsz).cuda(),
        ]
        self.net = nn.Sequential(*layers).cuda()

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x.float())
        x = self.net(x.float())
        if self.relu:
            x = F.relu(x.float(), inplace=True)
        return x  # (N, L, D)

#含缺失标签的二分类损失函数
def criterion(y, pred):
    fun = nn.BCEWithLogitsLoss(reduction="none")
    is_valid = y ** 2 > 0
    # Loss matrix
    loss_mat = fun(pred.double(), (y + 1) / 2)
    # loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

    loss = torch.sum(loss_mat) / torch.sum(is_valid)
    return loss

class FusionMamba(nn.Module):
    def __init__(self, args, num_tasks=1):
        super().__init__()

        self.args = args
        self.num_tasks = num_tasks

        n_input_proj = 2
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        #分子结构特征投影
        self.struct_proj = nn.Sequential(
            *[LinearLayer(2, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[0]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[1]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[2])][:n_input_proj])

        self.gnn_encoder = GNN(num_layer=args.gnn_layer, emb_dim=args.emb_dim).to(args.device)

        self.gnn_proj = nn.Sequential(
            *[LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[0]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[1]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[2])][:n_input_proj])


        self.graph_mamba_encoder = Graph_Mamba(d_model=args.d_model, n_layer=args.n_layer,
                                               sch_layer=args.sch_layer, dim_in=2, cutoff=args.cutoff).to(args.device)


        # 基础组件
        self.softmax = nn.Softmax(dim=1)
        self.frag_div_loss = nn.CrossEntropyLoss()
        self.frag_loss = nn.L1Loss()
        self.tree_loss = nn.CrossEntropyLoss()
        self.mask_loss = nn.MSELoss()  # nn.L1Loss()

        self.mlp = Sequential(
            Linear(args.d_model, args.d_model // 2),
            ReLU(),
            Dropout(p=args.drop),
            Linear(args.d_model // 2, args.d_model // 4),
            ReLU(),
            Dropout(p=args.drop),
            Linear(args.d_model // 4, num_tasks),
        )

        if args.use_fp:
            self.fp_mlp = Sequential(
                # Linear(2048 * 4 + 166 + 315, args.d_model * 2),
                Linear(2048 * 2 + 166 + 315, args.d_model * 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model * 2, args.d_model),
            )

        if args.use_gpt_embedding:
            self.gpt_mlp = Sequential(
                Linear(1536, args.d_model),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model, args.d_model),
            )

        if args.use_fg_embedding:
            self.fg_mlp = Sequential(
                Linear(133, args.d_model),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model, args.d_model),
            )



        if args.use_cat_fusion:
            if self.args.use_fp:
                self.final_mlp = Sequential(
                    Linear(args.d_model * 3, args.d_model),
                    ReLU(),
                    Dropout(p=args.drop),
                    Linear(args.d_model , args.d_model // 2),
                    ReLU(),
                    Dropout(p=args.drop),
                    Linear(args.d_model // 2, num_tasks),
                )
            else:
                self.final_mlp = Sequential(
                    Linear(args.d_model * 2, args.d_model),
                    ReLU(),
                    Dropout(p=args.drop),
                    Linear(args.d_model , args.d_model // 2),
                    ReLU(),
                    Dropout(p=args.drop),
                    Linear(args.d_model // 2, num_tasks),
                )
        else:
            self.final_mlp = Sequential(
                Linear(args.d_model, args.d_model // 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 2, args.d_model // 4),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 4, num_tasks),
            )


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        #从data中解析分子数据
        x, edge_index, edge_attr, gpt_embeddings, fg_embeddings  = data.x, data.edge_index, data.edge_attr, data.gpt_embeddings, data.fg_embeddings

        morgan_fps, morgan2048_fps, maccs_fps, rdkit_fps, atom_pair_fps, topological_fps, erg_fps \
            = data.morgan_fps, data.morgan2048_fps, data.maccs_fps, data.rdkit_fps, data.atom_pair_fps, data.topological_fps, data.erg_fps

        # fp_emb = torch.cat([morgan2048_fps, maccs_fps, rdkit_fps, atom_pair_fps, topological_fps, erg_fps], dim=-1)
        if self.args.use_fp:
            fp_emb = torch.cat([morgan2048_fps, maccs_fps, rdkit_fps, erg_fps], dim=-1)
            x_fp = self.fp_mlp(fp_emb)

        h = self.struct_proj(x)

        # initialization
        if self.args.use_gnn:
            x_g = self.gnn_encoder(x, edge_index, edge_attr)
            x_g = self.gnn_proj(x_g) + h
        else:
            x_g = h

        if self.args.use_gpt_embedding:
            gpt_embeddings = self.gpt_mlp(gpt_embeddings)
        else:
            gpt_embeddings = None

        if self.args.use_fg_embedding:
            fg_embeddings = self.fg_mlp(fg_embeddings)
        else:
            fg_embeddings = None

        x_m  = self.graph_mamba_encoder(data, gpt_emb=gpt_embeddings, fg_emb=fg_embeddings)

        x_gnn = global_mean_pool(x_g, data.node_batch)
        x_mam = global_mean_pool(x_m, data.node_batch)

        if self.args.use_cat_fusion:
            if self.args.use_fp:
                x_out = torch.cat((x_gnn, x_mam, x_fp), dim=-1)
            else:
                x_out = torch.cat((x_gnn, x_mam), dim=-1)
        else:
            if self.args.use_fp:
                x_out = x_gnn + x_mam + x_fp
            else:
                x_out = x_gnn + x_mam
        results = self.final_mlp(x_out)
        return results

