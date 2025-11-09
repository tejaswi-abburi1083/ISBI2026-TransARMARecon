# TransARMARecon: A Transductive ARMA-based Graph Neural Network with Sparse Labels for Neurodegenerative Diseases
## Overview: 
TransARMARecon is a framework for Alzheimer’s Disease (AD) classification using diffusion MRI data under a transductive learning setup.
The method constructs brain graphs based on region-wise connectivity features derived directly from diffusion MRI and employs histogram-based connectivity representations to capture the distribution of brain connectivity patterns.

High-dimensional diffusion MRI data are first projected into compact, lower-dimensional histogram-based features, which are then fed into the proposed TransARMARecon network.
The model leverages ARMA-based graph convolutional layers to achieve efficient signal reconstruction and robust classification, even with sparse label availability.
# Repository Structure
```
ISBI2026-TransARMARecon/
├── Classification_Scripts/
│ ├── CN_AD/
│ │ ├── GNN_Classifier_CN_AD/
│ │ │ ├── AE_GCN_Recon_CN_AD.ipynb
│ │ │ ├── ARMA_CN_AD.ipynb
│ │ │ ├── ARMA_Recon_CN_AD.ipynb
│ │ │ ├── CHEB_CN_AD.ipynb
│ │ │ ├── Cheb_Recon_CN_AD.ipynb
│ │ │ ├── GAT_CN_AD.ipynb
│ │ │ ├── GAT_Recon_CN_AD.ipynb
│ │ │ ├── GCN_CN_AD.ipynb
│ │ │ └── GCN_Recon_CN_AD.ipynb
│ │ └── Traditional_classifiers_CN_AD/
│ │ ├── MLP_Supervised_classification_CN_AD.ipynb
│ │ ├── Random_Forest_Supervised_classification_CN_AD.ipynb
│ │ ├── SVC_Supervised_classification_CN_AD.ipynb
│ │ └── XGboost_Supervised_classification_CN_AD.ipynb
│ ├── CN_MCI/
│ │ ├── GNN_Classifier_CN_MCI/
│ │ │ ├── AE_GCN_Recon_CN_MCI.ipynb
│ │ │ ├── ARMA_CN_MCI.ipynb
│ │ │ ├── ARMA_Recon_CN_MCI.ipynb
│ │ │ ├── CHEB_CN_MCI.ipynb
│ │ │ ├── Cheb_Recon_CN_MCI.ipynb
│ │ │ ├── GAT_CN_MCI.ipynb
│ │ │ ├── GAT_Recon_CN_MCI.ipynb
│ │ │ ├── GCN_CN_MCI.ipynb
│ │ │ └── GCN_Recon_CN_MCI.ipynb
│ │ └── Traditional_classifiers_CN_MCI/
│ │ ├── MLP_Supervised_classification_CN_MCI.ipynb
│ │ ├── Random_Forest_Supervised_classification_CN_MCI.ipynb
│ │ ├── SVC_Supervised_classification_CN_MCI.ipynb
│ │ └── XGboost_Supervised_classification_CN_MCI.ipynb
│ └── FTD/
│ ├── GNN_Classifiers_NIFD/
│ │ ├── AE_GCN_Recon_NIFD.ipynb
│ │ ├── ARMA_NIFD.ipynb
│ │ ├── ARMA_Recon_NIFD.ipynb
│ │ ├── CHEB_NIFD.ipynb
│ │ ├── Cheb_Recon_NIFD.ipynb
│ │ ├── GAT_NIFD.ipynb
│ │ ├── GAT_Recon_NIFD.ipynb
│ │ ├── GCN_NIFD.ipynb
│ │ └── GCN_Recon_NIFD.ipynb
│ └── Traditional_classifiers_NIFD/
│ ├── MLP_Supervised_classification_NIFD.ipynb
│ ├── Random_Forest_Supervised_classification_NIFD.ipynb
│ ├── SVC_Supervised_classification_NIFD.ipynb
│ └── XGboost_Supervised_classification_NIFD.ipynb
│
├── Example_Subjects_ADNI/
│ ├── 016_S_6839_F_70_AD/
│ ├── 033_S_10016_F_73_MCI/
│ └── 041_S_6354_F_76_CN/
│
├── Example_Subjects_NIFD/
│ ├── 1_S_0228_F_56_Controls/
│ └── 3_S_0012_M_65_Patient/
│
├── Histogram_Features_ADNI/
│ ├── Histogram_AD_FA_20bin_updated.npy
│ ├── Histogram_CN_FA_20bin_updated.npy
│ └── Histogram_MCI_FA_20bin_updated.npy
│
├── Histogram_Features_NIFD/
│ ├── NIFD_Control_FA_Histogram_Feature.npy
│ └── NIFD_Patients_FA_Histogram_Feature.npy
│
├── Preprocessing_Scripts/
│ ├── Step1_DICOM_to_NIFTI_conversion.py
│ ├── Step2_Finding_Quantitative_Parameters.py
│ ├── Step3_Generate_Diffusion_Tensor.py
│ ├── Step4_JHU_label_registration.py
│ ├── Step5_Adni_histogram_calculation.py
│ └── Step5_Histogram_Calculation_NIFD.ipynb
│
├── README.md
└── requirements.txt
```
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/tejaswi-abburi1083/ISBI2026-TransARMARecon.git
   cd ISBI2026-TransARMARecon
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Data

We use diffusion MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Due to data access restrictions, users need to apply for access to the ADNI database directly. An example subject with the expected data structure is provided in the directory.
## Acknowledgments

Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.

