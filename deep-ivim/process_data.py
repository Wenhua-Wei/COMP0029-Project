import os
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
from torch.utils.data import DataLoader
import config


def get_nifti_data(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def trim_background(scan_long, mask_long):
    x = len(mask_long)
    background_voxel_arr = []
    for idx in range(x):
        if mask_long[idx][0] == 0:
            background_voxel_arr.append(idx)
    background_voxel_arr = np.array(background_voxel_arr)
    scan_long = np.delete(scan_long, background_voxel_arr, 0)       
    return scan_long, background_voxel_arr

def trim_background2(scan_long, mask_long):
    # scan_long =  np.delete(scan_long, np.argwhere(mask_long == 0), 0)

    return scan_long[mask_long == 1, :]

def trim_0(scan_long_no_bg):
    x = len(scan_long_no_bg)
    Sb_0_idx_arr = []
    for idx in range(x):
        Sb_arr = scan_long_no_bg[idx]
        if Sb_arr[0] == 0.:
            Sb_0_idx_arr.append(idx)
    Sb_0_idx_arr = np.array(Sb_0_idx_arr)
    scan_long_no_bg = np.delete(scan_long_no_bg, Sb_0_idx_arr, 0)
    return scan_long_no_bg

def normalize_scan(scan_long, S0):
    dimension = scan_long.shape
    signal_layer_num = dimension[0]
    bval_num = dimension[1]
    for i in range(signal_layer_num):
        signal_layer = scan_long[i,:]
        for j in range(bval_num):
            if S0[i] == 0.:
                if signal_layer[j] == 0.:
                    normalized_signal = 0.
                else:
                    normalized_signal = 1.
            else:
                normalized_signal = signal_layer[j]/S0[i]
            # if normalized_signal
            scan_long[i,j] = normalized_signal
    return scan_long


def get_S0(scan_long, bval_list):
    b0_idx_arr = np.where(bval_list == 5)[0]
    b0_signal_mean = np.empty((0,1), float)
    for signal_arr in scan_long:
        signal_val_sum = 0
        for b0_idx in b0_idx_arr:
            signal_val = signal_arr[b0_idx]
            signal_val_sum += signal_val
        mean = signal_val_sum/len(b0_idx_arr)  
        mean = np.reshape(mean, (1,1))
        
        b0_signal_mean = np.append(b0_signal_mean, mean, axis=0) 

    return b0_signal_mean


if __name__ == '__main__':

    # S0 = np.load('/Users/weiwenhua/UGY4/COMP0029/COMP0029_Final_Year_Individual_Project/deep_noddi/S0.npy')
    # arr1 = []
    # arr2 = []
    # for i in range(len(S0)):
    #     if S0[i] == 0:
    #         arr1.append(i)
    #     if S0[i] > 0 and S0[i] < 1:
    #         arr2.append(i)
    # print(arr1)
    # print(arr2)


    scan = get_nifti_data(config.data_folder2 + '/100206/data.nii.gz')
    mask = get_nifti_data(config.data_folder2 + '/100206/nodif_brain_mask.nii.gz')
    scan_dimension = scan.shape
    x = scan_dimension[0]
    y = scan_dimension[1]
    z = scan_dimension[2]
    b = scan_dimension[3]
    scan_long = np.reshape(scan, (x*y*z, b))
    mask_long = np.reshape(mask, (x*y*z, 1))
    scan_long_no_background1 = trim_background2(scan_long, mask_long)
    scan_long_no_background, _ = trim_background(scan_long, mask_long)
    print(scan_long_no_background1.shape)
    print(scan_long_no_background)
    print(0)























































    # a = np.array([[[[0,0,0],[4,5,6]],[[7,8,9],[10,11,12]],[[13,8,9],[16,11,12]],[[7,19,5],[1,22,6]]],[[[10,5,93],[43,5,23]],[[70,5,54],[0,1,11]],[[13,8,48],[16,11,54]],[[37,1,32],[21,2,83]]]])
    # b = trim_background(a, 3)
    # print(b)

    # print(type(scan_long_no_background))

    # a = np.array([[0,1,5,3,4,0.2],[19,88,24,75,69,4]])
    # scan = normalize_scan2(a)
    # print(scan)

    # a = np.array([[2],[1],[1],[5],[1]])
    # b = np.array([1,2,3,4])
    # print(b*a)

    # s_1 = get_nifti_data(config.data_folder + '/T1w/Diffusion/grad_dev.nii.gz')


# import torch
# import torch.nn as nn

# class IVIMAutoencoder(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()

#         # Encoder
#         self.encoder_layers = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#         )

#         # IVIM model
#         self.ivim = nn.Linear(32, 3)

#         # Decoder
#         self.decoder_layers = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_size),
#         )

#     def forward(self, x):
#         # Encode the input
#         encoded = self.encoder_layers(x)

#         # Fit the IVIM model to the encoded input
#         ivim_params = self.ivim(encoded)

#         # Decode the IVIM parameters and output the reconstructed input
#         decoded = self.decoder_layers(ivim_params)
#         return decoded

# def normalize_scan(scan_long, S0):
#     dimension = scan_long.shape
#     signal_layer_num = dimension[0]
#     bval_num = dimension[1]
#     for i in range(signal_layer_num):
#         signal_layer = scan_long[i,:]
#         for j in range(bval_num):
#             normalized_signal = signal_layer[j]/S0[i]
#             # if normalized_signal
#             scan_long[i,j] = normalized_signal
#     return scan_long

# def min_max_normalize(scan_long):
#     voxel_num = scan_long.shape[0]
#     bval_num = scan_long.shape[1]
#     for i in range(voxel_num):
#         signal_val_arr = scan_long[i,:]
#         min = np.min(signal_val_arr)
#         max = np.max(signal_val_arr)     
#         for j in range(bval_num):
#             scan_long[i][j] = (scan_long[i][j]-min)/(max-min)

#     return scan_long

# np.argwhere(np.isnan(scan_long_no_background))
# # np.argwhere(np.isinf(S0))