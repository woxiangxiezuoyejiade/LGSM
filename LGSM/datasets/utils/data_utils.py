import torch
from torch_geometric.data import Data


def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    batch.smiles = []
    # keys follow node
    node_sum_keys = ["edge_index", "radius_edge_index"]  # radius_edge_index需要索引累加
    # keys follow frag
    # frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    # no_sum_keys = ["edge_attr",
    #                "x",
    #                "y",
    #                "pos",
    #                "map",
    #                "frag",
    #                "descriptors",
    #                "frag_unique"]
    no_sum_keys = ["edge_attr",
                   "edge_weight",  # 化学键的距离
                   "radius_edge_weight",  # radius_graph的距离（用于SchNet）
                   "x",
                   "pos",
                   "descriptors",
                   ]

    for key in node_sum_keys + no_sum_keys:
        batch[key] = []

    batch.y = []
    batch.gpt_embeddings = []
    # batch.gpt_embeddings_v1 = []
    batch.fg_embeddings = []

    batch.morgan_fps = []
    batch.morgan2048_fps = []
    batch.maccs_fps = []
    batch.rdkit_fps = []
    batch.atom_pair_fps = []
    batch.topological_fps = []
    batch.erg_fps = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    #节点索引累积偏移量
    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0

    for data in data_list:
        num_nodes = data.x.shape[0]

        # num_frags = data.frag.shape[0]

        batch.node_batch_size.append(num_nodes)

        # batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes,), i_node, dtype=torch.long))

        # batch.frag_batch.append(torch.full((num_frags,), i_frag, dtype=torch.long))

        batch.y.append(data.y)
        batch.gpt_embeddings.append(data.gpt_embedding)
        # batch.gpt_embeddings_v1.append(data.gpt_embedding_v1)
        batch.fg_embeddings.append(data.fg_embedding)

        batch.morgan_fps.append(data.morgan_fp)
        batch.morgan2048_fps.append(data.morgan2048_fp)
        batch.maccs_fps.append(data.maccs_fp)
        batch.rdkit_fps.append(data.rdkit_fp)
        batch.atom_pair_fps.append(data.atom_pair_fp)
        batch.topological_fps.append(data.topological_fp)
        batch.erg_fps.append(data.erg_fp)

        if hasattr(data, 'smiles'):
            batch.smiles.append(data.smiles)

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        # for key in frag_sum_keys:
        #     item = data[key]
        #     item = item + cumsum_frag
        #     batch[key].append(item)

        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        # cumsum_frag += num_frags
        # i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.pos = torch.cat(batch.pos, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    # 如果edge_weight存在，则拼接；否则创建一个空的张量
    if len(batch.edge_weight) > 0 and any(ew.numel() > 0 for ew in batch.edge_weight):
        batch.edge_weight = torch.cat(batch.edge_weight, dim=0)
    else:
        batch.edge_weight = torch.empty(0, dtype=torch.float32)
    # 拼接radius_edge_index和radius_edge_weight
    batch.radius_edge_index = torch.cat(batch.radius_edge_index, dim=-1)
    if len(batch.radius_edge_weight) > 0 and any(ew.numel() > 0 for ew in batch.radius_edge_weight):
        batch.radius_edge_weight = torch.cat(batch.radius_edge_weight, dim=0)
    else:
        batch.radius_edge_weight = torch.empty(0, dtype=torch.float32)
    batch.descriptors = torch.cat(batch.descriptors, dim=0)
    # batch.frag = torch.cat(batch.frag, dim=0)
    # batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    # batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    # batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    # batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    # batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    batch.y = torch.stack(batch.y)
    batch.gpt_embeddings = torch.stack(batch.gpt_embeddings)
    # batch.gpt_embeddings_v1 = torch.stack(batch.gpt_embeddings_v1)
    batch.fg_embeddings = torch.stack(batch.fg_embeddings)

    batch.morgan_fps = torch.stack(batch.morgan_fps)
    batch.morgan2048_fps = torch.stack(batch.morgan2048_fps)
    batch.maccs_fps = torch.stack(batch.maccs_fps)
    batch.rdkit_fps = torch.stack(batch.rdkit_fps)
    batch.atom_pair_fps = torch.stack(batch.atom_pair_fps)
    batch.topological_fps = torch.stack(batch.topological_fps)
    batch.erg_fps = torch.stack(batch.erg_fps)

    # batch.tree = torch.LongTensor([data.tree for data in data_list])

    return batch.contiguous()