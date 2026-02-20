### **Abstract**

Accurate molecular representation learning is fundamental to drug property prediction and plays a critical role in drug discovery and ADMET evaluation. Existing methods often rely on single-modality representations, such as molecular graphs or SMILES sequences, which limits their ability to capture complementary structural, geometric, and semantic information. In this work, we propose LGSM, a LLM-enhanced geometric learning framework for multimodal molecular representation. LGSM jointly integrates molecular fingerprints, 2D topological graphs, 3D geometric structures, functional group semantics, and large language model (LLM)–derived SMILES embeddings. Specifically, a SchNet–Graph-Mamba architecture is employed to model local and long-range 3D spatial dependencies, while cross-attention and FiLM mechanisms are introduced to inject functional group information and LLM-based semantic priors into structural modeling. Extensive experiments on four public ADMET benchmark datasets covering absorption, distribution, metabolism, and excretion tasks demonstrate that LGSM consistently outperforms state-of-the-art baseline methods under scaffold split settings. Furthermore, comprehensive ablation studies validate the contribution of each modality, and interpretability analyses combined with molecular docking experiments confirm that the model attends to chemically and biologically meaningful substructures. These results indicate that LGSM provides an effective, interpretable, and generalizable solution for molecular property prediction in drug discovery.

### **Approach**

![Model](img.png)

### **Env**

**Step 1**:Install PyTorch with the appropriate CUDA version for your system. The command below is provided as an example for CUDA 11.8; please visit the PyTorch official website to obtain the installation command that matches your hardware configuration:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**Step 2**:Step 2: Install PyTorch Geometric and its dependencies:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
```
**Step 3**:Install all remaining dependencies:
```bash
pip install -r requirements.txt
```


### **Data Preparation**
All datasets used in this study are sourced from Therapeutics Data Commons (TDC). The raw data files have been organized and uploaded to the data folder of this repository for direct use.

### **Data Loading and Preprocessing**
The loader_downstream.py script located in the datasets folder is responsible for loading and preprocessing the raw data. Upon execution, the script will automatically create a processed subfolder within the corresponding dataset directory, where all preprocessed data files will be saved for subsequent model training and evaluation.
Run the following commands from the project root directory to complete the preprocessing step:

```
cd datasets

python loader_downstream.py
```

### **Model Training**
This project provides separate configuration files and training scripts for classification and regression tasks respectively:

**Classification task:** configuration file config.py, training script train_class.py

**Regression task:** configuration file config_reg.py, training script train_spearman.py

Before training, please modify the relevant hyperparameters (e.g., learning rate, batch size, number of training epochs) in the corresponding configuration file according to your requirements. Once the configuration is complete, run the appropriate training script to start the training process.

### **Training**

```
python train_class.py

python train_spearman.py
```
