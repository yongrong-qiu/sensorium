import numpy as np
import torch
from torch import nn
import torch.optim as optim
import tqdm
import copy
    

class BinaryDecoder(nn.Module):
    """
    Binary decoding model with dynamically created linear layers based on input_dim and latent_dims_list.
    """
    def __init__(self, input_dim=6611, latent_dims_list=[100, 100]):
        super().__init__()
        
        # Create a list to hold all the layers
        self.layers = nn.ModuleList()
        
        # Add the first layer (input_dim -> first latent dimension)
        self.layers.append(nn.Linear(input_dim, latent_dims_list[0]))
        self.layers.append(nn.ReLU())
        
        # Add intermediate layers (latent_dim[i] -> latent_dim[i+1])
        if len(latent_dims_list)>1:
            for ii in range(len(latent_dims_list) - 1):
                self.layers.append(nn.Linear(latent_dims_list[ii], latent_dims_list[ii+1]))
                self.layers.append(nn.ReLU())
        
        # Add the final output layer (last latent dimension -> 1)
        self.layers.append(nn.Linear(latent_dims_list[-1], 1))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        # Pass the input through all layers
        for layer in self.layers:
            x = layer(x)
        return x
    
 
def model_train(
    model, 
    X_train, # torch tensor
    y_train, # torch tensor
    X_val, # torch tensor
    y_val, # torch tensor
    device,
    n_epochs=200, # # number of epochs to run
    batch_size=10,  # size of each batch
    lr=0.0001, # learning rate
    verbose=False, # not display the progress
):
    """
    Model training
    """

    model = model.to(device)

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_start = np.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    acc_vals = []
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size].float().to(device)
                y_batch = y_train[start:start+batch_size].float().to(device)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                # acc = (y_pred.round() == y_batch).float().mean().detach().cpu().data.numpy()
                # bar.set_postfix(
                #     loss=float(loss).detach().cpu().data.numpy(),
                #     acc=acc
                # )
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.float().to(device)
        y_val = y_val.float().to(device)
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean().detach().cpu().data.numpy()
        acc_vals.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc, best_weights, acc_vals