import os
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
from torch.utils.data import DataLoader
import config
import random


def get_nifti_data(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def remove_background(scan_long, mask_long):
    is_bg = mask_long == 1
    is_bg = is_bg.reshape(len(is_bg))
    return scan_long[is_bg,:]

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
    bval_5_indices = np.where(bval_list == 0.005)[0]
    scan_bval_5 = scan_long[:, bval_5_indices]
    mean_bval_5 = np.mean(scan_bval_5, axis=1)
    mean_bval_5 = mean_bval_5[:, np.newaxis]
    return mean_bval_5

# bvals should be in 1e3
def rand_select_b_group_indices(b_values, n):
    b5_indices = np.where(b_values == 5.)[0]
    num_groups = len(b5_indices)
    rand_selected_groups = random.sample(range(num_groups), n)
    rand_selected_b5_indices = b5_indices[rand_selected_groups]
    print()
    return rand_selected_b5_indices

def get_rand_selected_bval_indice(b_values, n):
    b5_indices = np.where(b_values == 5.)[0]
    rand_selected_b5_indices = rand_select_b_group_indices(b_values, n)
    group_ranges = []
    for index, start in enumerate(b5_indices):
        if start in rand_selected_b5_indices:
            if index + 1 == len(b5_indices):
                group_ranges.append((start, len(b_values)))
            else:
                group_ranges.append((start, b5_indices[index+1]))

    rand_selected_bval_indice = [list(range(start, end)) for start, end in group_ranges]
    flat_rand_selected_bval_indice = [item for sublist in rand_selected_bval_indice for item in sublist]
    return flat_rand_selected_bval_indice

def get_rand_subset_bvals(b_values, n):
    rand_selected_bval_indice = get_rand_selected_bval_indice(b_values, n)
    return b_values[rand_selected_bval_indice]

def get_mask_pro(scan, mask):
    scan_dimension = scan.shape
    x = scan_dimension[0]
    y = scan_dimension[1]
    z = scan_dimension[2]
    b = scan_dimension[3]

    scan_long = np.reshape(scan, (x*y*z, b))
    mask_long = np.reshape(mask, (x*y*z, 1))

    indice_beginning_0_all = np.where(scan_long[:, 0] == 0)[0]

    mask_long_pro = np.copy(mask_long)
    mask_long_pro[indice_beginning_0_all] = 0

    return mask_long_pro


def add_bg(mask_long, params):
    resume_params = np.copy(mask_long)
    no_bg_indices = np.where(mask_long == 1)[0]
    for i, index in enumerate(no_bg_indices):
        resume_params[index] = params[i]
    return resume_params

def back_to_3D(mask_long, parames_1d, shape):
    params_with_bg = add_bg(mask_long, parames_1d.detach().numpy())
    params_3d = np.reshape(params_with_bg, shape)
    return params_3d


# if __name__ == "__main__":
#     bvals_all_100206 = np.loadtxt(config.data_folder2 + '/100206/bvals')
#     rand_selected_bval_indice = get_rand_selected_bval_indice(bvals_all_100206, 9)
#     print()




















































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

# def trim_background(scan_long, mask_long):
#     x = len(mask_long)
#     background_voxel_arr = []
#     for idx in range(x):
#         if mask_long[idx][0] == 0:
#             background_voxel_arr.append(idx)
#     background_voxel_arr = np.array(background_voxel_arr)
#     scan_long = np.delete(scan_long, background_voxel_arr, 0)       
#     return scan_long, background_voxel_arr

# def trim_0(scan_long_no_bg):
#     x = len(scan_long_no_bg)
#     Sb_0_idx_arr = []
#     for idx in range(x):
#         Sb_arr = scan_long_no_bg[idx]
#         if Sb_arr[0] == 0.:
#             Sb_0_idx_arr.append(idx)
#     Sb_0_idx_arr = np.array(Sb_0_idx_arr)
#     scan_long_no_bg = np.delete(scan_long_no_bg, Sb_0_idx_arr, 0)
#     return scan_long_no_bg