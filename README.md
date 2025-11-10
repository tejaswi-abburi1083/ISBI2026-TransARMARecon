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
│ └── Step5_nifd_registration_histogram_calculation.ipynb
├── JHU-ICBM-labels-1mm.nii.gz   ← required for ROI-based feature extraction
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
We use diffusion MRI data from two publicly available neuroimaging databases:
Alzheimer’s Disease Neuroimaging Initiative (ADNI): Used for Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects. Due to data access restrictions, users must apply for access directly at [adni.loni.usc.edu](https://adni.loni.usc.edu/)

Neuroimaging in Frontotemporal Dementia (NIFD): Used for Control and Patient subjects. The dataset can be accessed through the [NIFD database](https://ida.loni.usc.edu/login.jsp) upon registration and approval.

Example subjects with the expected directory structure are provided in the Example_Subjects_ADNI/ and Example_Subjects_NIFD/ folders.
#### Note
Before executing any classification or reconstruction scripts, please ensure that the Fractional Anisotropy (FA) histogram feature paths are correctly specified in your code.

Pre-extracted FA histogram features for all subject categories are provided within the repository for reference:
```
├── Histogram_Features_ADNI/
│   ├── Histogram_AD_FA_20bin_updated.npy
│   ├── Histogram_CN_FA_20bin_updated.npy
│   └── Histogram_MCI_FA_20bin_updated.npy
│
├── Histogram_Features_NIFD/
│   ├── NIFD_Control_FA_Histogram_Feature.npy
│   └── NIFD_Patients_FA_Histogram_Feature.npy
```
If you intend to utilize these provided example features, please update the corresponding file paths in your training or evaluation scripts.
For example:
```
fa_path = "./Histogram_Features_ADNI/Histogram_CN_FA_20bin_updated.npy"
```
## Atlas Acknowledgment
The atlas used for the study is JHU-ICBM-labels-1mm.nii.gz and has been derived from the JHU ICBM-DTI-81 White-Matter Labels Atlas created by the Laboratory of Brain Anatomical MRI, Johns Hopkins University.

#### Citation:

Mori, S., Oishi, K., Faria, A. V., & van Zijl, P. C. M. (2008). MRI Atlas of Human White Matter (2nd Edition). Academic Press. The atlas is distributed as part of the FMRIB Software Library (FSL) and is also available via the NeuroImaging Tools & Resources Collaboratory (NITRC).
#### Website:
```
https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
```
## Dataset Acknowledgment
Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.
