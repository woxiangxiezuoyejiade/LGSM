import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
import os
os.sys.path.append('./')
# os.sys.path.append('../')
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit.Chem import AllChem, rdMolDescriptors, rdReducedGraphs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat, product, chain

from datasets.fragment.mol_bpe import Tokenizer
from datasets import mol_to_graph_data_obj_pos, create_standardized_mol_id
import logging

organic_major_ish = {'[C]', '[O]', '[N]', '[F]', '[Cl]', '[Br]', '[I]', '[S]', '[P]', '[B]', '[H]'}

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from datasets.utils.logger import setup_logger

project_root = os.path.dirname(os.path.dirname(__file__))
print("project_root",project_root)
GLOBAL_DATA_ROOT = os.path.join(project_root, "data", "adme")

log_file_abs_path = os.path.join(project_root, "data_log.txt")
# 传入绝对路径，第二个参数传日志所在目录（从绝对路径中提取），filename传文件名
logger = setup_logger('MSE.Data', os.path.dirname(log_file_abs_path), filename=os.path.basename(log_file_abs_path))#logger = setup_logger('MSE.Data', './', filename="data_log.txt")
logger.info('-'*40)
logger.info("Running the loader_downstream.py")
# 定义要计算的分子描述符列表
descriptor_names = [desc[0] for desc in Descriptors._descList]

# 创建分子描述符计算器
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


TASKS = {
    'VDss_Lombardo': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
}

def load_tasks_from_csv(dataset_name):
    dataset_raw_dir = os.path.join(GLOBAL_DATA_ROOT, dataset_name, 'raw')
    csv_files = [f for f in os.listdir(dataset_raw_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"在 {dataset_raw_dir} 下未找到 CSV 文件")

    input_path = os.path.join(dataset_raw_dir, csv_files[0])
    input_df = pd.read_csv(input_path, sep=',')

    tasks = list(input_df.columns)[1:]
    del input_df
    return tasks

def load_all_tasks():
    try:
        if 'sider' not in TASKS or not TASKS['sider']:
            TASKS['sider'] = load_tasks_from_csv('sider')
    except FileNotFoundError:
        logger.warning("sider CSV not found, skipping.")

    try:
        if 'toxcast' not in TASKS or not TASKS['toxcast']:
            TASKS['toxcast'] = load_tasks_from_csv('toxcast')
    except FileNotFoundError:
        logger.warning("toxcast CSV not found, skipping.")


class MoleculeDataset(InMemoryDataset):
    def __init__(self, dataset, root, vocab_file_path="./datasets/vocab.txt"):
        self.dataset = dataset
        self.root = root
        if self.dataset not in TASKS or not TASKS[self.dataset]:
            TASKS[self.dataset] = load_tasks_from_csv(self.dataset)
        self.vocab_file_path = vocab_file_path
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)
            self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        self.root = root

        super().__init__(self.root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        #beginning
        processed_file = self.processed_paths[0]
        try:
            # 尝试加载 processed 数据
            self.data, self.slices = torch.load(processed_file, weights_only=False)
        except Exception as e:
            print(f"[Warning] Failed to load processed file {processed_file}: {e}")
            print("[Info] Removing corrupted processed file and regenerating...")

            # 删除坏的 processed 文件
            if os.path.exists(processed_file):
                os.remove(processed_file)

            # 重新处理数据
            self.process()
            self.data, self.slices = torch.load(processed_file, weights_only=False)
            #end
        self.transform, self.pre_transform, self.pre_filter = None, None, None

        # 若当前processed文件缺少半径属性，或调用方希望刷新，则按当前cutoff补写
        radius_missing = not (hasattr(self.data, 'radius_edge_index') and hasattr(self.data, 'radius_edge_weight') and
                              'radius_edge_index' in self.slices and 'radius_edge_weight' in self.slices)
        if radius_missing:
            try:
                print(f"[Info] radius attributes missing for {self.dataset}/{self.split}, generating with cutoff={self.cutoff} ...")
                add_radius_attrs_to_file(processed_file, cutoff=self.cutoff)
                # 重新加载
                self.data, self.slices = torch.load(processed_file, weights_only=False)
            except Exception as e:
                print(f"[Warning] Auto-generate radius attributes failed: {e}")

    def process(self):
        data_smiles_list = []
        data_list = []
        logger.info(f"Expected save path for {self.dataset}: {self.processed_paths[0]}")
        '''from utils.splitters import scaffold_split
        downstream_dir = [
            'dataset/adme/CYP2C9_Substrate_CarbonMangels',
            'dataset/adme/Half_Life_Obach',
            'dataset/adme/VDss_Lombardo',
            'dataset/adme/Pgp_Broccatelli'
        ]'''
        # 通用下游数据集仅返回 (smiles_list, rdkit_mol_objs, labels)
        # 个别数据集（如 bace）若需要 folds，请在此单独处理
        loaded = load_dataset(self.dataset, self.raw_paths[0])
        if len(loaded) == 4:
            smiles_list, rdkit_mol_objs, labels, folds = loaded
        else:
            smiles_list, rdkit_mol_objs, labels = loaded
            folds = None
        for i in tqdm(range(len(smiles_list))):
            #调试
            print(f"Processing molecule {i}/{len(smiles_list)}")
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol == None:
                continue
            smiles = smiles_list[i]

            # 调试
            print(f"Running mol_to_graph_data_obj_pos for {i}")
            data = mol_to_graph_data_obj_pos(rdkit_mol,smiles)
            print(f"Finished molecule {i}")

            # descriptors-----------------------------------------
            descriptors = calculator.CalcDescriptors(rdkit_mol)
            data.descriptors = torch.Tensor(descriptors)
            data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
            data.y = torch.tensor(labels)
            if self.dataset == 'bace' and folds is not None:
                data.fold = torch.tensor([folds[i]])

            try:
                tree = self.tokenizer(smiles)
            except:
                logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                continue

            # Manually consructing the fragment graph
            map = [0] * data.num_nodes
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[], []]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                        # extend frag
                        frag[node_i][0] = self.vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src, dst])
                    frag_edge_index[1].extend([dst, src])
            except KeyError as e:
                logger.info(f"Line {i}, Error in matching subgraphs {e}")
                continue

            unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            data.map = torch.LongTensor(map)  #原子到片段节点的映射
            data.frag = torch.LongTensor(frag) # 片段节点的token ID
            data.frag_edge_index = torch.LongTensor(frag_edge_index) # 片段之间的边连接
            data.frag_unique = frag_unique

            data_list.append(data)

            data_smiles_list.append(smiles_list[i])

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
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

            # 统一数据格式
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

            # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_morgan_fingerprint(mol, radius=2):
    """get morgan fingerprint"""
    nBits = 200
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return [int(b) for b in mfp.ToBitString()]

def get_morgan2048_fingerprint(mol, radius=2):
    """get morgan2048 fingerprint"""
    nBits = 2048
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return [int(b) for b in mfp.ToBitString()]

def get_maccs_fingerprint(mol):
    """get maccs fingerprint"""
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()[1:]]

def get_rdkit_fingerprint(mol):
    """get rdkit fingerprint"""
    fp = AllChem.RDKFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def get_atom_pair_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> list[int]:
    """
    生成原子对指纹。
    它编码了分子中所有原子对（按其类型）以及它们之间的成键距离。
    对于捕捉分子的长程相互作用和骨架信息很有用。

    Args:
        mol (Chem.Mol): RDKit分子对象。
        n_bits (int): 指纹的长度。

    Returns:
        list[int]: 长度为 n_bits 的0/1列表。
    """
    if mol is None:
        return [0] * n_bits
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return [int(b) for b in fp.ToBitString()]

def get_topological_torsion_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> list[int]:
    """
    生成拓扑扭转指纹。
    它编码了由四个连续键合原子组成的每个线性序列（扭转角）。
    对分子的构象灵活性和骨架分支模式敏感。

    Args:
        mol (Chem.Mol): RDKit分子对象。
        n_bits (int): 指纹的长度。

    Returns:
        list[int]: 长度为 n_bits 的0/1列表。
    """
    if mol is None:
        return [0] * n_bits
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    return [int(b) for b in fp.ToBitString()]

def get_erg_fingerprint(mol: Chem.Mol) -> list[float]:
    """
    生成ERG (Extended Reduced Graph) 指纹。
    这是一种基于药效团的指纹，它将分子简化为功能性节点图，
    并编码节点间的路径。

    注意：这个指纹的输出是一个数值向量（浮点数列表），而不是二进制向量。

    Args:
        mol (Chem.Mol): RDKit分子对象。

    Returns:
        list[float]: 长度为315的浮点数列表。
    """
    if mol is None:
        return [0.0] * 315
    # GetErGFingerprint 返回一个元组，我们将其转换为列表
    fp = rdReducedGraphs.GetErGFingerprint(mol)
    return list(fp)

class DownstreamMoleculeDataset(InMemoryDataset):
    def __init__(self, dataset, root, split, vocab_file_path="./datasets/vocab.txt", cutoff=None):
        self.dataset = dataset
        self.root = root
        self.split = split
        with open(os.path.join(self.root, 'raw', f'{self.dataset}.pkl'), 'rb') as f:
            self.gpt_embedding = pickle.load(f)

        self.fg2emb = pickle.load(open('./fg2emb.pkl', 'rb'))

        with open('./funcgroup.txt', "r") as f:
            funcgroups = f.read().strip().split('\n')
            name = [i.split()[0] for i in funcgroups]
            self.smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
            self.smart2name = dict(zip(self.smart, name))

        
        # 如果没有提供cutoff，尝试从config导入，否则使用默认值10.0
        if cutoff is None:
            try:
                from config import cfg
                self.cutoff = cfg.cutoff
            except (ImportError, AttributeError):
                # 如果无法导入config，使用默认值10.0
                self.cutoff = 10.0
        else:
            self.cutoff = cutoff  # 使用传入的cutoff值

        self.vocab_file_path = vocab_file_path
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)
            self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        self.root = root
        super().__init__(self.root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f'{self.split}_data_processed.pt'), weights_only=False)

        #beginning
        processed_file = self.processed_paths[0]
        try:
            # 尝试加载 processed 数据
            self.data, self.slices = torch.load(processed_file, weights_only=False)
        except Exception as e:
            print(f"[Warning] Failed to load processed file {processed_file}: {e}")
            print("[Info] Removing corrupted processed file and regenerating...")

            # 删除坏的 processed 文件
            if os.path.exists(processed_file):
                os.remove(processed_file)

            # 重新处理数据
            self.process()
            self.data, self.slices = torch.load(processed_file, weights_only=False)
            #end
        self.transform, self.pre_transform, self.pre_filter = None, None, None

    @property
    def raw_file_names(self):
        # file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return f'{self.split}.csv'

    @property
    def processed_file_names(self):
        return f'{self.split}_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate val  id location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        logger.info(f"Expected save path for {self.dataset}: {self.processed_paths[0]}")

        smiles_list, rdkit_mol_objs, labels = load_dataset(self.dataset,self.raw_paths[0])
        for i in tqdm(range(len(smiles_list))):
            #调试
            print(f"Processing molecule {i}/{len(smiles_list)}")
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol == None:
                continue
            smiles = smiles_list[i]

            fg_emb = []  #收集分子中存在的官能团
            pad_fg = [[0] * 133]
            name_list = []
            # print(fg_emb)
            for sm in self.smart:
                # print(Chem.MolToSmiles(sm))
                if rdkit_mol.HasSubstructMatch(sm):
                    # print(len(fg2emb[smart2name[sm]].tolist()))
                    fg_emb.append(self.fg2emb[self.smart2name[sm]].tolist())
                    name_list.append(self.smart2name[sm])
            if len(fg_emb) > 12:
                fg_emb = fg_emb[:12]
            else:
                fg_emb.extend(pad_fg * (12 - len(fg_emb)))

            # 调试
            print(f"Running mol_to_graph_data_obj_pos for {i}")
            data = mol_to_graph_data_obj_pos(rdkit_mol, smiles)
            print(f"Finished molecule {i}")

            # 计算并保存edge_weight（化学键的距离）
            # edge_weight是一个列表，第i个元素表示第i条化学键的长度
            if data.edge_index.numel() > 0:
                row, col = data.edge_index
                edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                data.edge_weight = edge_weight.float()  # 确保是float类型
            else:
                # 如果没有边，创建一个空的edge_weight
                data.edge_weight = torch.empty(0, dtype=torch.float32)

            # 计算并保存radius_graph的edge_index和edge_weight（用于SchNet和Graph_Mamba）
            # 这样可以避免训练时重复计算radius_graph和距离
            if data.pos.numel() > 0:
                # 使用配置中的cutoff值计算radius_graph
                radius_edge_index = radius_graph(data.pos, r=self.cutoff, batch=None)
                if radius_edge_index.numel() > 0:
                    row_r, col_r = radius_edge_index
                    radius_edge_weight = (data.pos[row_r] - data.pos[col_r]).norm(dim=-1)
                    data.radius_edge_index = radius_edge_index
                    data.radius_edge_weight = radius_edge_weight.float()
                else:
                    # 如果没有边，创建空的张量
                    data.radius_edge_index = torch.empty((2, 0), dtype=torch.long)
                    data.radius_edge_weight = torch.empty(0, dtype=torch.float32)
            else:
                data.radius_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.radius_edge_weight = torch.empty(0, dtype=torch.float32)

            # descriptors-----------------------------------------
            descriptors = calculator.CalcDescriptors(rdkit_mol)
            data.descriptors = torch.Tensor(descriptors)
            data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
            data.y = torch.tensor(labels[i])
            data.gpt_embedding = torch.tensor(self.gpt_embedding[smiles]['smiles_embeddings'], dtype=torch.float32)
            # data.gpt_embedding_v1 = torch.tensor(self.gpt_embedding[smiles]['V1_prompt_smiles_embeddings'], dtype=torch.float32)
            data.fg_embedding = torch.tensor(fg_emb, dtype=torch.float32)

            data.morgan_fp = torch.tensor(get_morgan_fingerprint(rdkit_mol), dtype=torch.float32)  #200
            data.morgan2048_fp = torch.tensor(get_morgan2048_fingerprint(rdkit_mol), dtype=torch.float32)  #2048
            data.maccs_fp = torch.tensor(get_maccs_fingerprint(rdkit_mol), dtype=torch.float32)  #166
            data.rdkit_fp = torch.tensor(get_rdkit_fingerprint(rdkit_mol), dtype=torch.float32)  #2048
            data.atom_pair_fp = torch.tensor(get_atom_pair_fingerprint(rdkit_mol), dtype=torch.float32) # 2048
            data.topological_fp = torch.tensor(get_topological_torsion_fingerprint(rdkit_mol), dtype=torch.float32) #2048
            data.erg_fp = torch.tensor(get_erg_fingerprint(rdkit_mol), dtype=torch.float32) #315

            data_list.append(data)

            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

            # 统一数据格式
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

            # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def add_radius_attrs_to_file(processed_file, cutoff=None):
    """
    给已处理的.pt文件添加 radius_edge_index 和 radius_edge_weight 属性
    用于更新旧版本处理的数据，新数据会在process()中自动包含这些属性
    
    Args:
        processed_file: .pt文件路径
        cutoff: radius_graph的截断半径，如果为None则从config读取
    """
    if cutoff is None:
        try:
            from config import cfg
            cutoff = cfg.cutoff
        except (ImportError, AttributeError):
            cutoff = 10.0
    
    if not os.path.exists(processed_file):
        print(f"文件不存在: {processed_file}")
        return
    
    print(f"加载文件: {processed_file}")
    data, slices = torch.load(processed_file, weights_only=False)
    
    # 检查是否已有这些属性
    if hasattr(data, 'radius_edge_index') and hasattr(data, 'radius_edge_weight'):
        if 'radius_edge_index' in slices and 'radius_edge_weight' in slices:
            print("属性已存在，跳过")
            return
    
    print(f"开始添加属性（cutoff={cutoff}）...")
    
    num_samples = len(slices['x']) - 1
    radius_edge_index_list = []
    radius_edge_weight_list = []
    
    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i+1}/{num_samples}")
        
        # 获取当前样本的节点范围
        node_start = slices['x'][i]
        node_end = slices['x'][i + 1]
        pos_sample = data.pos[node_start:node_end]
        
        # 计算radius_graph
        if pos_sample.numel() > 0:
            radius_edge_index = radius_graph(pos_sample, r=cutoff, batch=None)
            if radius_edge_index.numel() > 0:
                row_r, col_r = radius_edge_index
                radius_edge_weight = (pos_sample[row_r] - pos_sample[col_r]).norm(dim=-1)
                # 调整边索引（加上节点偏移）
                radius_edge_index = radius_edge_index + node_start
                radius_edge_index_list.append(radius_edge_index)
                radius_edge_weight_list.append(radius_edge_weight.float())
            else:
                radius_edge_index_list.append(torch.empty((2, 0), dtype=torch.long))
                radius_edge_weight_list.append(torch.empty(0, dtype=torch.float32))
        else:
            radius_edge_index_list.append(torch.empty((2, 0), dtype=torch.long))
            radius_edge_weight_list.append(torch.empty(0, dtype=torch.float32))
    
    # 拼接并添加到data
    data.radius_edge_index = torch.cat(radius_edge_index_list, dim=-1)
    data.radius_edge_weight = torch.cat(radius_edge_weight_list, dim=0)
    
    # 更新slices
    slices['radius_edge_index'] = [0]
    slices['radius_edge_weight'] = [0]
    cumsum_edge = 0
    cumsum_weight = 0
    for i in range(num_samples):
        cumsum_edge += radius_edge_index_list[i].shape[1]
        cumsum_weight += radius_edge_weight_list[i].shape[0]
        slices['radius_edge_index'].append(cumsum_edge)
        slices['radius_edge_weight'].append(cumsum_weight)
    
    # 保存
    print(f"保存到: {processed_file}")
    torch.save((data, slices), processed_file)
    print(f"完成！添加了 {num_samples} 个样本的属性")


def add_radius_attrs_to_datasets(datasets=None, data_dir='./data/adme', cutoff=None):
    """
    批量给多个数据集的.pt文件添加 radius_edge_index 和 radius_edge_weight 属性
    
    Args:
        datasets: 数据集名称列表，如果为None则自动检测data_dir下的所有数据集
        data_dir: 数据目录
        cutoff: radius_graph的截断半径，如果为None则从config读取
    """
    if cutoff is None:
        try:
            from config import cfg
            cutoff = cfg.cutoff
        except (ImportError, AttributeError):
            cutoff = 10.0
    
    # 如果没有指定数据集，自动检测
    if datasets is None:
        if not os.path.exists(data_dir):
            print(f"数据目录不存在: {data_dir}")
            return
        
        # 获取所有数据集目录
        datasets = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                # 检查是否有processed目录
                processed_path = os.path.join(item_path, 'processed')
                if os.path.exists(processed_path):
                    datasets.append(item)
        
        print(f"自动检测到 {len(datasets)} 个数据集: {datasets}")
    
    # 处理每个数据集
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        splits = ['train', 'valid', 'test']
        processed_count = 0
        skipped_count = 0
        
        for split in splits:
            file_path = os.path.join(data_dir, dataset_name, 'processed', f'{split}_data_processed.pt')
            if os.path.exists(file_path):
                print(f"\n处理 {split} split...")
                try:
                    # 检查是否已有属性
                    data, slices = torch.load(file_path, weights_only=False)
                    if (hasattr(data, 'radius_edge_index') and hasattr(data, 'radius_edge_weight') and
                        'radius_edge_index' in slices and 'radius_edge_weight' in slices):
                        print(f"  ✓ {split}: 属性已存在，跳过")
                        skipped_count += 1
                    else:
                        add_radius_attrs_to_file(file_path, cutoff)
                        processed_count += 1
                except Exception as e:
                    print(f"  ✗ {split}: 处理失败 - {e}")
            else:
                print(f"  ⚠ {split}: 文件不存在，跳过")
                skipped_count += 1
        
        print(f"\n数据集 {dataset_name} 处理完成:")
        print(f"  新增属性: {processed_count} 个split")
        print(f"  已跳过: {skipped_count} 个split")


# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain
def merge_dataset_objs(dataset_1, dataset_2):
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    #扩展标签，统一标签维度 新标签标记为0
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

#分子指纹
def create_circular_fingerprint(mol, radius, size, chirality):

    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)

class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality

        self._load()

    def _process(self):
        data_smiles_list = []
        data_list = []
        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, 'processed_fp')
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False,
                                  header=False)
        with open(os.path.join(processed_dir,
                               'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(processed_dir,
                                   'fingerprint_data_processed.pkl'),
                      'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ## if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(self.root, self.dataset, self.radius, self.size,
                                                 chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


def load_dataset(dataset_name, input_path):
    dataset_name = dataset_name.lower()
    input_df = pd.read_csv(input_path, sep=',')


    smiles_col = 'Drug' if 'Drug' in input_df.columns else 'smiles'
    smiles_list = input_df[smiles_col]
    rdkit_mol_objs_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [Chem.MolToSmiles(m) if m is not None else None for m in preprocessed_rdkit_mol_objs_list]
    invalid_count = sum(1 for m in preprocessed_rdkit_mol_objs_list if m is None)
    logger.info(f"Invalid SMILES count: {invalid_count}")

    # if isinstance(TASKS[dataset_name], list):
    #     labels = input_df[TASKS[dataset_name]]
    # else:
    #     labels = input_df[[TASKS[dataset_name]]]

    labels = input_df['Y']

    return smiles_list, preprocessed_rdkit_mol_objs_list, labels.values


def _load_chembl_with_labels_dataset(root_path):
    # 1. load folds and labels
    f = open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array(
        [np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    logger.info('preprocessing')
    for i in tqdm(range(len(rdkitArr))):
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]  # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_all_datasets():
    downstream_dir = [
        'Pgp_Broccatelli',
        'CYP2C9_Substrate_CarbonMangels',
        'Half_Life_Obach',
        'VDss_Lombardo'
    ]

    vocab_file_path = os.path.join(os.path.dirname(__file__), "vocab.txt") #debug时使用
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(f'{project_root}', "data", "adme")
    for dataset_name in downstream_dir:
        logger.info(dataset_name)
        root = os.path.join(data_root, dataset_name)
        os.makedirs(root + "/processed", exist_ok=True)
        #dataset = MoleculeDataset(dataset=dataset_name, root=root)  #命令行运行时使用
        dataset = MoleculeDataset(dataset=dataset_name,root=root,vocab_file_path=vocab_file_path) #debug时使用 搭配1761行
        logger.info(dataset)


# test MoleculeDataset object
if __name__ == "__main__":
    import argparse
    
    # 检查是否是命令行调用add_radius_attrs功能
    parser = argparse.ArgumentParser(description='数据处理工具')
    parser.add_argument('--add_radius_attrs', action='store_true', help='给.pt文件添加radius属性')
    parser.add_argument('--file', type=str, help='.pt文件路径')
    parser.add_argument('--dataset', type=str, nargs='+', help='数据集名称（可指定多个，用空格分隔）')
    parser.add_argument('--all_datasets', action='store_true', help='处理data_dir下的所有数据集')
    parser.add_argument('--split', type=str, default='train', help='split名称（train/valid/test），仅当指定--file时使用')
    parser.add_argument('--data_dir', type=str, default='./data/adme', help='数据目录')
    parser.add_argument('--cutoff', type=float, default=None, help='cutoff值（默认从config读取）')
    
    args, unknown = parser.parse_known_args()
    
    if args.add_radius_attrs:
        # 添加radius属性的命令行工具
        if args.file:
            # 处理单个文件
            file_path = args.file
            add_radius_attrs_to_file(file_path, args.cutoff)
        elif args.all_datasets:
            # 处理所有数据集
            add_radius_attrs_to_datasets(datasets=None, data_dir=args.data_dir, cutoff=args.cutoff)
        elif args.dataset:
            # 处理指定的数据集（可以是多个）
            add_radius_attrs_to_datasets(datasets=args.dataset, data_dir=args.data_dir, cutoff=args.cutoff)
        else:
            print("必须指定以下选项之一:")
            print("  --file: 处理单个文件")
            print("  --dataset: 处理指定的数据集（可指定多个）")
            print("  --all_datasets: 处理所有数据集")
            parser.print_help()
    else:
        create_all_datasets()