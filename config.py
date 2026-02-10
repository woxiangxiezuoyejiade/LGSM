import torch
from dataclasses import dataclass, asdict
# from datasets.utils.logger import setup_logger, dict_to_markdown
import yaml
from datetime import datetime


# AUROC
#dataset = 'Pgp_Broccatelli'
## AUPRC
dataset = 'CYP2C9_Substrate_CarbonMangels'


task_type = 'classification'
split = 'scaffold'  # 'scaffold' 'random'
dim = 256


@dataclass
class ModelArgs:
    # task
    dataset: str = dataset
    split: str = split
    task_type: str = task_type
    data_dir: str = './data/adme/'
    model: str = 'FusionMamba'
    pretrain_path: str = None

    # training settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    runseed: int = 56711
    seed: int = 42
    num_workers: int = 8
    epoch: int = 100
    batchsize: int = 8
    lr: float = 0.0001
    decay: float = 0.0
    drop: int = 0.2

    # module settings
    use_gnn: bool = True
    use_graph_mamba: bool = True
    use_fp: bool = True
    use_gpt_embedding: bool = True
    use_fg_embedding: bool = True
    use_cat_fusion: bool = True


    # gnn encoder settings
    gnn_layer: int = 6  # 6
    emb_dim: int = 128

    # graph_mamba settings
    d_model: int =  128 #128
    n_layer: int = 2
    sch_layer: int = 6
    dim_in: int = 2
    cutoff: float = 10.0


    # loss weight
    w_task: float = 1.0

    save_model: bool = True
    # save_dir = ''
    #train
    save_dir = f"./result_{task_type}/{dataset}/{split}/{dataset}_{runseed}_{lr}_{datetime.now().strftime('%b%d_%H:%M:%S')}/"

    def to_dict(self):
        return asdict(self)

    def save_yaml(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


cfg = ModelArgs()

