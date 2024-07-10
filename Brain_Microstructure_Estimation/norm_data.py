import numpy as np

import process_data
import argparse

def norm(dmri_path, mask_path, bval_path):
    dmri = process_data.get_nifti_data(dmri_path)
    mask = process_data.get_nifti_data(mask_path)
    dmri_dimension = dmri.shape
    x = dmri_dimension[0]
    y = dmri_dimension[1]
    z = dmri_dimension[2]
    b = dmri_dimension[3]
    dmri_long = np.reshape(dmri, (x*y*z, b))
    mask_long = np.reshape(mask, (x*y*z, 1))

    # Exclude error voxels which have zero signal intensities at b=5
    indices_beginning_0_all = np.where(dmri_long[:, 0] == 0)[0]
    # Let mask_pro includes error voxels as background as well
    mask_long_pro = np.copy(mask_long)
    mask_long_pro[indices_beginning_0_all] = 0
    # Remove all background voxels from dMRI
    dmri_long_no_background = process_data.remove_background(dmri_long, mask_long_pro)
    bvals = np.loadtxt(bval_path) * 1e-3
    S0 = process_data.get_S0(dmri_long_no_background, bvals)
    # Normalize the signal intensities in DWIs by dividing the corresponding S0
    normalized_dmri = dmri_long_no_background / S0
    # Save the normalized and processed dMRI
    np.save("normalized_dmri.npy",normalized_dmri)

    return normalized_dmri

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and preprocess dMRI data")
    parser.add_argument('--dmri_path', type=str, required=True, help='Path to the dMRI data in .nii.gz format')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask in .nii.gz format')
    parser.add_argument('--bval_path', type=str, required=True, help='Path to the file containing b values')

    args = parser.parse_args()

    normalized_dmri = norm(args.dmri_path, args.mask_path, args.bval_path)
