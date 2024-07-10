This repository contains the code for training and evaluating a self-supervised artificial neural network (ANN) on diffusion MRI (dMRI) data. The ANN is designed to model the ball-and-stick model parameters from dMRI data, including $\theta$, $\phi$, $\lambda_{\parallel}$, $\lambda_{\text{iso}}$, and volume fraction.

## Dataset
Six dMRI datasets from the Human Connectome Project (HCP) are used in this project, including one training set, one validation set, and four testing sets. The original HCP data can be found [here](https://drive.google.com/drive/folders/1okSWyHuj0WJSLG3mlAQo70RoVnEsF2Uc?usp=sharing), and the normalized and processed HCP data can be found [here](https://drive.google.com/drive/folders/1wzgFoZyexkBL40GtGDcNit-WRiHJA96I?usp=sharing).

## Getting Started

* Create and activate a virtual environment using Conda:
conda create --name myenv
conda activate myenv

* Install the required packages from the COMP0029_CLBF0/requirements.txt file:
conda install --file requirements.txt

## Data Preprocessing

To use the normalized and processed HCP data directly for model training, download the files [here](https://drive.google.com/drive/folders/1wzgFoZyexkBL40GtGDcNit-WRiHJA96I?usp=sharing).
Alternatively, you can normalize and process the data yourself by running the following command:
python norm_data.py --dmri_path path1 --mask_path path2 --bval_path path3
where:
path1: path to a data.nii.gz file in the Original HCP folder
path2: path to a nodif_brain_mask.nii.gz file in the Original HCP folder
path3: path to a bvals file in the Original HCP folder

## Training
To train the self-supervised ANN for ball-and-stick model fitting, run the following command in the terminal:
python main.py --trainset_path trainset_dir_path --valset_path valset_dir_path --m_per_shell num
where:
trainset_dir_path: path to the normalized training data folder
valset_dir_path: path to the normalized validation data folder
num: number of measurements **in each shell** the user would like to choose (e.g., set num=90 for full measurements, num=45 for half measurements)

## Evaluation
A few pre-trained ANN models are stored in the COMP0029_CLBF0/saved_ANN directory. To apply a trained ANN model on unseen data and evaluate its performance, refer to the evaluation.ipynb notebook. 