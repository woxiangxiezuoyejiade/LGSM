#### **Abstract**

Accurate molecular representation learning is fundamental to drug property prediction and plays a critical role in drug discovery and ADMET evaluation. Existing methods often rely on single-modality representations, such as molecular graphs or SMILES sequences, which limits their ability to capture complementary structural, geometric, and semantic information. In this work, we propose LGSM, a LLM-enhanced geometric learning framework for multimodal molecular representation. LGSM jointly integrates molecular fingerprints, 2D topological graphs, 3D geometric structures, functional group semantics, and large language model (LLM)–derived SMILES embeddings. Specifically, a SchNet–Graph-Mamba architecture is employed to model local and long-range 3D spatial dependencies, while cross-attention and FiLM mechanisms are introduced to inject functional group information and LLM-based semantic priors into structural modeling. Extensive experiments on four public ADMET benchmark datasets covering absorption, distribution, metabolism, and excretion tasks demonstrate that LGSM consistently outperforms state-of-the-art baseline methods under scaffold split settings. Furthermore, comprehensive ablation studies validate the contribution of each modality, and interpretability analyses combined with molecular docking experiments confirm that the model attends to chemically and biologically meaningful substructures. These results indicate that LGSM provides an effective, interpretable, and generalizable solution for molecular property prediction in drug discovery.

#### **Approach**

![Model](img.png)

#### **Env**

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

pip install -r requirements.txt

#### **Training**

python train_class.py

python train_spearman.py
