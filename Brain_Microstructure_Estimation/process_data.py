import nibabel as nib
import numpy as np
import random

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# Load a dMRI data in nnii.gz format
def get_nifti_data(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

# Function used to remove the background from the dMRI data 
# using the correspnding mask that indicates the background regions.
def remove_background(scan_long, mask_long):
    is_bg = mask_long == 1
    is_bg = is_bg.reshape(len(is_bg))
    return scan_long[is_bg,:]

# Normalized the signal intensities S(g, b) in DWIs by divding the corresponding S0
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
            scan_long[i,j] = normalized_signal
    return scan_long

# Get the SO of each voxel in DWIs from dMRI
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

    return rand_selected_b5_indices

# Consider voxels with very low signal intensities at b=5 as background in order to eliminate them.
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

# Convert estimated parameters to a 3D map
def back_to_3D(mask_long, parames_1d, shape):
    params_with_bg = add_bg(mask_long, parames_1d.detach().numpy())
    params_3d = np.reshape(params_with_bg, shape)
    return params_3d

# Save the loss values during training
def save_loss(file_path, train_loss_list, avg_train_loss_list, val_loss_list):
    with open(file_path, 'w') as f:
        f.write('train_loss_per_epoch_list: ' + str(train_loss_list) + '\n')
        f.write('\n')
        f.write('avg_train_loss_list: ' + str(avg_train_loss_list) + '\n')
        f.write('\n')
        f.write('val_loss_list: ' + str(val_loss_list) + '\n')

# Creates an acquisition scheme object from bvalues, gradient directions,
# pulse duration $\delta$ and pulse separation time $\Delta$.
def get_acquisition_scheme(bvals, gradient_directions):
    delta = 0.0106
    Delta = 0.0431
    return acquisition_scheme_from_bvalues(bvals, gradient_directions, delta, Delta)

# Computes measurement indices for each b-shell using the Dmipy acquisition scheme
def get_shelled_measurement_indices(bvals, bvecs):
    scheme = get_acquisition_scheme(bvals*1e6, bvecs.T)
    b_shells = scheme.shell_bvalues * 1e-6
    distance = scheme.min_b_shell_distance * 1e-6

    shelled_measurement_indices = {}
    for shell in b_shells:
        shell_range = (shell - distance, shell + distance)
        shelled_measurement_indices[shell] = np.where((bvals >= shell_range[0]) & (bvals <= shell_range[1]))[0]

    return shelled_measurement_indices


# Selects a specified number of gradient direction points from a given set of points, 
# ensuring a more uniform distribution by maximizing the minimum distance between the selected points.
def farthest_point_sampling(points, num_samples):
    if num_samples > len(points):
        raise ValueError("Number of samples must be less than or equal to the number of points")

    selected_indices = [np.random.randint(len(points))]  # Randomly select the first point
    distances = np.full(len(points), np.inf)

    for _ in range(num_samples - 1):
        new_distances = np.linalg.norm(points - points[selected_indices[-1]], axis=1)
        distances = np.minimum(distances, new_distances)
        farthest_point_index = np.argmax(distances)
        selected_indices.append(farthest_point_index)

    return selected_indices


# Selects a specified number of gradient directions from each b-shell group, 
# ensuring a more uniform distribution across the gradient directions in each group 
# by using the farthest_point_sampling method.
def select_uniform_gradient_directions(grad_dir_groups, num_samples_per_group):
    uniform_selected_indices = {}

    for shell_val in grad_dir_groups:
        grad_dir = grad_dir_groups[shell_val]
        selected_indices = farthest_point_sampling(grad_dir, num_samples_per_group)
        uniform_selected_indices[shell_val] = selected_indices

    return uniform_selected_indices

# Returns a dictionary containing the gradient directions grouped by b-shells.
def get_shelled_bvecs(bvecs, shelled_measurement_indices):
    grad_dir_groups = {}
    for shell_val in shelled_measurement_indices.keys():
        if shell_val <= 10.:
            continue
        shell_indices = shelled_measurement_indices[shell_val]
        shell_bvecs = bvecs[:,shell_indices].T
        grad_dir_groups[shell_val] = shell_bvecs
    return grad_dir_groups

# Obtains a sorted list of measurement indices for selected gradient directions across b-shells
def get_grad_dir_indices(uniform_selected_indices, shelled_measurement_indices):
    result = []
    for shell_val in shelled_measurement_indices.keys():
        if shell_val < 10.:
            continue
        idx = uniform_selected_indices[shell_val]
        grad_dir_indices = shelled_measurement_indices[shell_val][idx]
        result.extend(grad_dir_indices)
    return sorted(result)
