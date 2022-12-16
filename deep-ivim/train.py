import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import config
import data_load

class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 3))

    def forward(self, X):
        params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)

        X = Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt)

        return X, Dp, Dt, Fp


if __name__ == '__main__':

    bval_list = np.loadtxt(config.data_folder2 + '/100206/bvals')
    normalized_scan_long_no_background = np.load('normalized_scan_long.npy')

    print(normalized_scan_long_no_background)

    b_values = torch.FloatTensor(bval_list)
    net = Net(b_values)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)  

    trainloader, num_batches = data_load.get_train_loader(normalized_scan_long_no_background)

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 10

    # Train
    for epoch in range(1): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # print('X_batch: ')
            # print(X_batch.shape)
            # print(X_batch)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
            # print('X_pred: ')
            # print(X_pred.shape)
            # print(X_pred)
            # print(int(X_pred[0,0]))
            loss = criterion(X_pred, X_batch)
            loss.backward()
            # print(loss)
            optimizer.step()
            running_loss += loss.item()
            # print('Running loss: ', running_loss)
            if torch.isnan(torch.tensor(running_loss)):
            # if running_loss == np.nan:
                test=0
            # if i == 50:
            #     break

        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    # Restore best model
    net.load_state_dict(final_model)