import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
from tqdm import tqdm
from torch_geometric.data import Data
# import os
# os.sys.path.append('.')
# os.sys.path.append('../')
from fragment.mol_bpe import Tokenizer
from datasets import mol_to_graph_data_obj_pos


organic_major_ish = {'[C]', '[O]', '[N]', '[F]', '[Cl]', '[Br]', '[I]', '[S]', '[P]', '[B]', '[H]'}

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

# 定义要计算的分子描述符列表
descriptor_names = [desc[0] for desc in Descriptors._descList]

# 创建分子描述符计算器
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


class MoleculePretrainDataset(InMemoryDataset):
    def __init__(self, root, smiles_column=None, data_file_path=None, vocab_file_path=None):

        self.smiles_column = smiles_column
        self.data_file_path = data_file_path
        self.vocab_file_path = vocab_file_path
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)    #在这里传入的vocab路径 然后间接影响到了mol_bpe.具体路径写在了本文件最后
            self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        self.folder = root   #将root的值赋给属性folder 这是一个自定义的属性

        super(MoleculePretrainDataset, self).__init__(self.folder, transform=None, pre_transform=None)

        #self.data, self.slices = torch.load(self.processed_paths[0])  9月27为解决pdb调试时出现pyg安全报错而修改
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    #处理smiles字符串
    def process(self):
        import pdb;
        pdb.set_trace()
        data_df = pd.read_csv(self.data_file_path)[:200]

        if not (self.smiles_column in data_df.columns):
            raise ValueError("The specified SMILES column name is not found in the data file.")

        if data_df.isnull().values.any():
            raise ValueError("Missing values found in the data file.")

        tasks = [column for column in data_df.columns if column != self.smiles_column] #任务列
        smiles_list = data_df[self.smiles_column]
        task_list = data_df[tasks]

        data_list = []
        for i in tqdm(range(len(smiles_list))):

            # data = Data()

            smiles = smiles_list[i]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            # task_labels = task_list.iloc[i].values.astype(np.float32)
            data = mol_to_graph_data_obj_pos(mol) #转为图数据对象

            #计算分子描述符 并转为Tensor张量
            descriptors = calculator.CalcDescriptors(mol)
            data.descriptors = torch.Tensor(descriptors)

            # data.y = None #torch.Tensor(task_labels)
            try:
                tree = self.tokenizer(smiles)
            except:
                print("Unable to process SMILES:", smiles)
                continue

            # Manually consructing the fragment graph 构建分子片段
            map = [0] * data.num_nodes
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[], []]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map  map[i]是第i个原子所属的片段编号
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                        # extend frag frag[i]该片段对应的词表编号
                        frag[node_i][0] = self.vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src, dst])
                    frag_edge_index[1].extend([dst, src])
            except KeyError as e:
                print("Error in matching subgraphs", e)
                continue

            unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            #将上面的信息存入图数据data中
            data.map = torch.LongTensor(map)
            data.frag = torch.LongTensor(frag)
            data.frag_edge_index = torch.LongTensor(frag_edge_index)
            data.frag_unique = frag_unique

            data_list.append(data)

        #对片段图进行哈希处理
        tree_dict = {}
        hash_str_list = []
        for data in data_list:
            tree = Data()
            tree.x = data.frag
            tree.edge_index = data.frag_edge_index
            nx_graph = to_networkx(tree, to_undirected=True)
            hash_str = weisfeiler_lehman_graph_hash(nx_graph)
            if hash_str not in tree_dict:
                tree_dict[hash_str] = len(tree_dict)
            hash_str_list.append(hash_str)

        tree = []
        for hash_str in hash_str_list:
            tree.append(tree_dict[hash_str])

        for i, data in enumerate(data_list):
            data.tree = tree[i]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])



def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index"]
    # no sum keys 无需累加的键
    no_sum_keys = ["edge_attr",
                   "x",
                   "pos",
                   "map",
                   "frag",
                   "descriptors",
                   "frag_unique"]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    # batch.y = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0  #用于边索引
    i_node = 0

    cumsum_frag = 0 #用于片段索引
    i_frag = 0

    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]

        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes,), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags,), i_frag, dtype=torch.long))

        # batch.y.append(data.y)  索引偏移且为累加偏移

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)

        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    #特征拼接
    batch.x = torch.cat(batch.x, dim=0)
    batch.pos = torch.cat(batch.pos, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.descriptors = torch.cat(batch.descriptors, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    # batch.y = torch.stack(batch.y)

    batch.tree = torch.LongTensor([data.tree for data in data_list])

    return batch.contiguous()

import argparse

if __name__ == '__main__':
    root = './data/adme/Pgp_Broccatelli'
    data_file_path = './data/adme/Pgp_Broccatelli/raw/valid.csv'
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_vocab_path = os.path.join(script_dir, 'vocab.txt')

    parser = argparse.ArgumentParser(description='Prepare molecular data')
    parser.add_argument('--root', type=str, default=root, help='Path to the data folder')
    parser.add_argument('--data_file_path', default=data_file_path, type=str, help='If creation, path to raw data')
    parser.add_argument('--smiles_column', default='smiles', type=str, help='Name of the colum containing smiles in the raw data table')
    parser.add_argument('--vocab_file_path', default=default_vocab_path, type=str)
    args = parser.parse_args()

    dataset = MoleculePretrainDataset(root=args.root,
                                      smiles_column=args.smiles_column,
                                      data_file_path=args.data_file_path,
                                      vocab_file_path=args.vocab_file_path)