import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

import process_data
import network
import trainer

import uuid
import argparse


# Prepares data for ANN input
def prepare_data(normalized_scan_full, bvals_all, bvecs_all, subset_indices):
    normalized_scan_sub = normalized_scan_full[:, subset_indices]
    bvals_sub = bvals_all[subset_indices]*1e-3
    bvecs_sub = bvecs_all[:, subset_indices]

    big_b_indice_sub = np.where(bvals_sub != 0.005)[0]
    normalized_scan_sub_no_b5 = normalized_scan_sub[:, big_b_indice_sub]
    bvals_sub_no_b5 = bvals_sub[big_b_indice_sub]
    bvecs_sub_no_b5 = bvecs_sub[:,big_b_indice_sub].T

    return normalized_scan_sub_no_b5, bvals_sub_no_b5, bvecs_sub_no_b5

# Creates a PyTorch DataLoader with a specified batch size for the input data.
def get_data_loader(normalized_data):
    batch_size = 128
    dataloader = utils.DataLoader(torch.from_numpy(normalized_data.astype(np.float32)),
                            batch_size = batch_size, 
                            shuffle = True,
                            num_workers = 2,
                            drop_last = True)
    return dataloader

'''
The main function that loads and preprocesses the data, selects subset of measurements if required, 
trains a neural network (BallStickNet) to predict ball-and-stick model parameters, 
and saves the trained model and loss information.
'''
def run(trainset_path, valset_path, m_per_shell):
    run_id = str(uuid.uuid4())
    log_filename = f'logfile_{run_id}.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Load training data
    normalized_scan_trainset_full = np.load(trainset_path + '/normalized_scan_100206_full.npy')
    bvals_all_trainset = np.loadtxt(trainset_path + '/100206_bvals')
    bvecs_all_trainset = np.loadtxt(trainset_path + '/100206_bvecs')
    # Load validation data
    normalized_scan_valset_full = np.load(valset_path + '/normalized_scan_100408_full.npy')
    bvals_all_valset = np.loadtxt(valset_path + '/100408_bvals')
    bvecs_all_valset = np.loadtxt(valset_path + '/100408_bvecs')

    # Computes the indices of the measurements for each b-value shell
    shelled_measurement_indices = process_data.get_shelled_measurement_indices(bvals_all_trainset, bvecs_all_trainset)
    # Groups the gradient directions for each b-value shell
    grad_dir_groups = process_data.get_shelled_bvecs(bvecs_all_trainset, shelled_measurement_indices)
    # Selects a specified number of gradient directions from each b-value shell group 
    # while maintaining a uniform distribution across the gradient directions 
    uniform_selected_indices = process_data.select_uniform_gradient_directions(grad_dir_groups, m_per_shell)
    # Obtains a sorted list of measurement indices across b-shells
    selected_m_indices = process_data.get_grad_dir_indices(uniform_selected_indices, shelled_measurement_indices)
    
    num_measurements = m_per_shell*3
    logging.info(f"Num of measurement groups: {num_measurements}")
    
    subset_indices = selected_m_indices
    if num_measurements != 270:
        sub_filename = f'rand_sub_{run_id}.txt'
        with open(sub_filename, 'w') as f:
            f.write(str(subset_indices))
    
    # Prepare train and validation data for ANN
    normalized_scan_trainset_sub_no_b5, bvals_trainset_sub_no_b5, bvecs_trainset_sub_no_b5 = prepare_data(normalized_scan_trainset_full, bvals_all_trainset, bvecs_all_trainset, subset_indices)
    normalized_scan_valset_sub_no_b5, bvals_valset_sub_no_b5, bvecs_valset_sub_no_b5 = prepare_data(normalized_scan_valset_full, bvals_all_valset, bvecs_all_valset, subset_indices)

    b_values_trainset_sub_no_b5 = torch.FloatTensor(bvals_trainset_sub_no_b5)
    grad_dir_trainset_sub_no_b5 = torch.FloatTensor(bvecs_trainset_sub_no_b5)

    b_values_valset_sub_no_b5 = torch.FloatTensor(bvals_valset_sub_no_b5)
    grad_dir_valset_sub_no_b5 = torch.FloatTensor(bvecs_valset_sub_no_b5)

    trainloader = get_data_loader(normalized_scan_trainset_sub_no_b5)
    valset = torch.from_numpy(normalized_scan_valset_sub_no_b5.astype(np.float32))

    num_batches = len(normalized_scan_trainset_sub_no_b5) // 128

    logging.info(f"The shape of trainset: {normalized_scan_trainset_sub_no_b5.shape}")
    logging.info(f"The shape of valset: {normalized_scan_valset_sub_no_b5.shape}")

    net = network.BallStickNet(grad_dir_trainset_sub_no_b5, b_values_trainset_sub_no_b5, device)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.00001)

    # Self-supervised guided ANN training
    final_model, train_loss_list, avg_train_loss_list, val_loss_list = trainer.train(device, net, trainloader, valset, optimizer, criterion, num_batches, grad_dir_valset_sub_no_b5, b_values_valset_sub_no_b5, logging)
  
    # Saves the best ANN model with the lowest validation loss upon training completion
    model_filename = f'model_{run_id}.pt'
    torch.save(final_model, model_filename)
    
    # Saves the loss values during training
    loss_filename = f'loss_{run_id}.txt'
    process_data.save_loss(loss_filename, train_loss_list, avg_train_loss_list, val_loss_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and validation.")
    parser.add_argument('--trainset_path', type=str, required=True, help='Path to the training set')
    parser.add_argument('--valset_path', type=str, required=True, help='Path to the validation set')
    parser.add_argument('--m_per_shell', type=int, required=True, help='Number of samples per shell')

    args = parser.parse_args()

    run(args.trainset_path, args.valset_path, args.m_per_shell)
    
    # run('/Users/weiwenhua/UGY4/COMP0029/Submission/COMP0029_CLBF0/trainset',  '/Users/weiwenhua/UGY4/COMP0029/Submission/COMP0029_CLBF0/valset', 90)