### **Abstract**

Accurate prediction of absorption, distribution, metabolism, and excretion (ADMET) properties is critical for lead optimization and early risk assessment in drug discovery. However, many existing computational approaches rely on a single molecular representation, limiting their ability to simultaneously capture topological features, three-dimensional conformational information, and chemically meaningful functional group characteristics that underlie structure–pharmacokinetic relationships. Here, we present a multimodal molecular representation framework that integrates two-dimensional topology, three-dimensional conformational descriptors, molecular fingerprints, and functional group–level semantic features. A large language model–based encoding of SMILES sequences was further incorporated to enhance contextual structural representation. Through hierarchical fusion of structural and semantic information, the model captures both local chemical environments and global spatial relationships within a unified framework. The proposed approach was evaluated on four publicly available benchmark datasets spanning absorption, distribution, metabolism, and excretion tasks under a scaffold split setting to assess generalization to structurally novel compounds. Compared with representative graph neural network models, fingerprint-based methods, and automated machine learning approaches, the framework consistently achieved superior predictive performance. Moreover, interpretability analysis combined with molecular docking studies demonstrated that the model reliably identifies key substructures associated with transporter recognition, metabolic liability, and distribution-related behavior. Overall, this multimodal strategy provides accurate and interpretable predictions, offering a practical computational tool to support rational ADMET evaluation and lead optimization in early-stage drug design.

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
