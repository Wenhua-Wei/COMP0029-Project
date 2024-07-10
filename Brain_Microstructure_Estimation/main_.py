import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import process_data
import network
import trainer
import uuid
import argparse
import os

def prepare_data(normalized_scan_full, bvals_all, bvecs_all, rand_sub_bval_indice):
    if rand_sub_bval_indice == None:
        normalized_scan_sub = normalized_scan_full
        bvals_sub = bvals_all*1e-3
        bvecs_sub = bvecs_all
    else:
        normalized_scan_sub = normalized_scan_full[:, rand_sub_bval_indice]
        bvals_sub = bvals_all[rand_sub_bval_indice]*1e-3
        bvecs_sub = bvecs_all[:, rand_sub_bval_indice]

    big_b_indice_sub = np.where(bvals_sub != 0.005)[0]
    normalized_scan_sub_no_b5 = normalized_scan_sub[:, big_b_indice_sub]
    bvals_sub_no_b5 = bvals_sub[big_b_indice_sub]
    bvecs_sub_no_b5 = bvecs_sub[:,big_b_indice_sub].T

    return normalized_scan_sub_no_b5, bvals_sub_no_b5, bvecs_sub_no_b5

def get_data_loader(normalized_data):
    batch_size = 128
    dataloader = utils.DataLoader(torch.from_numpy(normalized_data.astype(np.float32)),
                            batch_size = batch_size, 
                            shuffle = True,
                            num_workers = 2,
                            drop_last = True)
    return dataloader

def run(rand_sub_bval_indice, trainset_dir_path, valset_dir_path):
    run_id = str(uuid.uuid4())
    log_filename = f'logfile_{run_id}.log'
    # log_filename = 'logfile_5g.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    normalized_scan_trainset_full = np.load(trainset_dir_path + '/normalized_scan_100206_full.npy')
    bvals_all_trainset = np.loadtxt(trainset_dir_path + '/100206_bvals')
    bvecs_all_trainset = np.loadtxt(trainset_dir_path + '/100206_bvecs')

    normalized_scan_valset_full = np.load(valset_dir_path + '/normalized_scan_100408_full.npy')
    bvals_all_valset = np.loadtxt(valset_dir_path + '/100408_bvals')
    bvecs_all_valset = np.loadtxt(valset_dir_path + '/100408_bvecs')

    # arg 1 
    '''
    Output to a txt
    '''
    if rand_sub_bval_indice == None:
        num_m = 270
    else:
        num_m = len(rand_sub_bval_indice)
    logging.info(f"Num of measurements: {num_m}")
    # rand_sub_bval_indice = np.array(process_data.get_rand_selected_bval_indice(bvals_all_trainset, num_groups))
    # rand_sub_str = np.array2string(rand_sub_bval_indice, separator=',')
    # sub_filename = f'rand_sub_{run_id}.txt'
    # # sub_filename = 'rand_sub.txt'
    # with open(sub_filename, 'w') as f:
    #     f.write(rand_sub_str)

    # rand_sub_bval_indice = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
    # rand_sub_bval_indice = []
    # for i in range(288):
    #     rand_sub_bval_indice.append(i)
    
    if num_m != 270:
        sub_filename = f'rand_sub_{run_id}.txt'
        with open(sub_filename, 'w') as f:
            f.write(str(rand_sub_bval_indice))

    normalized_scan_trainset_sub_no_b5, bvals_trainset_sub_no_b5, bvecs_trainset_sub_no_b5 = prepare_data(normalized_scan_trainset_full, bvals_all_trainset, bvecs_all_trainset, rand_sub_bval_indice)
    normalized_scan_valset_sub_no_b5, bvals_valset_sub_no_b5, bvecs_valset_sub_no_b5 = prepare_data(normalized_scan_valset_full, bvals_all_valset, bvecs_all_valset, rand_sub_bval_indice)

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

    final_model, train_loss_list, avg_train_loss_list, val_loss_list = trainer.train(device, net, trainloader, valset, optimizer, criterion, num_batches, grad_dir_valset_sub_no_b5, b_values_valset_sub_no_b5, logging)
    # arg 2
    model_filename = f'model_{run_id}.pt'
    # model_filename = 'model_5g.pt'
    torch.save(final_model, model_filename)
    # arg 3
    loss_filename = f'loss_{run_id}.txt'
    # loss_filename = 'loss_5g.txt'
    process_data.save_loss(loss_filename, train_loss_list, avg_train_loss_list, val_loss_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for the BallStickNet model")
    parser.add_argument("--rand_sub_bval_indice", type=int, nargs='*', default=None, help="Random sub bval indice")
    parser.add_argument("--base_path", type=str, default="/Users/weiwenhua/UGY4/COMP0029/Submission/COMP0029_CLBF0", help="Base path for dataset directories")
    
    return parser.parse_args()

def read_indices_from_file(file_path):
    with open(file_path, 'r') as f:
        indices = [int(index.strip()) for index in f.readlines()]
    return indices

if __name__ == "__main__":
    args = parse_arguments()
    if args.rand_sub_bval_indice_file:
        rand_sub_bval_indice = read_indices_from_file(args.rand_sub_bval_indice_file)
    else:
        rand_sub_bval_indice = None
    
    run(None, '/Users/weiwenhua/UGY4/COMP0029/Submission/COMP0029_CLBF0/trainset',  '/Users/weiwenhua/UGY4/COMP0029/Submission/COMP0029_CLBF0/valset')