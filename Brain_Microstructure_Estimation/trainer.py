import torch

import network

def train(device, net, trainloader, val_set, optimizer, criterion, num_batches, gradient_directions_val_set, b_values_val_set, logging):

    train_loss_list=[]
    avg_train_loss_list=[]
    avg_val_loss_list = []


    best_train_loss = 1e16
    best_val_loss = 1e16

    num_bad_epochs_train = 0
    num_bad_epochs_val = 0

    patience = 15

    for epoch in range(1000):
        logging.info("-----------------------------------------------------------------")
        logging.info("Epoch: {}; Bad training epochs: {}; Bad validation epochs: {}".format(epoch, num_bad_epochs_train, num_bad_epochs_val))

        net.train()
        running_train_loss = 0.

        running_val_loss = 0.
    
        #     Training
        for i, X_batch in enumerate(trainloader, 0):
            # Perform a forward pass through the network, obtaining ball-and-stick model parameters prediction
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            X_pred, theta, phi, mu_cart_pred, lambda_par_pred, lambda_iso_pred, volume_0_pred, volume_1_pred = net(X_batch)
            X_pred = X_pred.to(device)
            # Compute the loss between the synthesize and the actual signal intensities in DWIs.
            loss = criterion(X_pred.type(torch.FloatTensor), X_batch.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        train_loss_list.append(running_train_loss)
        avg_train_loss_list.append(running_train_loss/num_batches)
        logging.info("Loss: {}; Average Loss: {}".format(running_train_loss, running_train_loss/num_batches))
    
        #     Validation
        temp_model = net.state_dict()
        val_net = network.BallStickNet(gradient_directions_val_set, b_values_val_set, device)
        val_net.load_state_dict(temp_model)
        val_net = val_net.to(device)
        val_net.eval()
        with torch.no_grad():
            val_set = val_set.to(device)
            val_pred, _, _, _, _, _, _, _ = val_net(val_set)
            val_pred = val_pred.to(device)
            val_loss = criterion(val_pred.type(torch.FloatTensor), val_set.type(torch.FloatTensor))
            running_val_loss = val_loss.item()
            avg_val_loss_list.append(running_val_loss)
            logging.info("Validation Loss: {}".format(running_val_loss))
    
        if running_train_loss < best_train_loss:
            best_train_loss = running_train_loss
            num_bad_epochs_train = 0
        
        elif running_train_loss >= best_train_loss:
            num_bad_epochs_train = num_bad_epochs_train + 1

        # Save the ANN model each time there is a reduction in validation loss.
        if running_val_loss < best_val_loss:
            logging.info("############### Saving good model for validation set ###############################")
            final_model = net.state_dict()
            best_val_loss = running_val_loss
            num_bad_epochs_val = 0
        elif running_val_loss >= best_val_loss:
            num_bad_epochs_val += 1
            # Stop training when no reduction in validation loss for 15 epochs
            if num_bad_epochs_val == patience:
                logging.info("Done, best training loss per epoch: {}; best validation loss: {}".format(best_train_loss, best_val_loss))
                break

    logging.info("Done")
    net.load_state_dict(final_model)
    

    return final_model, train_loss_list, avg_train_loss_list, avg_val_loss_list