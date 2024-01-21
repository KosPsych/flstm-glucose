from itertools import zip_longest
import torch
import numpy as np
import torch.nn as nn

import copy

class EarlyStopper:
    """
    Implements early stopping for training neural networks.
    Monitors a validation metric to stop training if no improvement.

    Attributes:
    - patience (int): Specified patience value.
    - min_delta (float): Specified min_delta value.
    - counter (int): Counts epochs with no improvement.
    - min_validation_loss (float): Minimum validation loss. Initialized to infinity.
    - best_model_state: The best model found so far (lowest validation loss)

    Methods:
    - early_stop(validation_loss): Returns True if stopping criteria are met; otherwise, False.
    """

    def __init__(self, patience=2, min_delta=0):
        """Initializes an EarlyStopper instance."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state = None

    def early_stop(self, validation_loss, model_state):
        """Returns True if stopping criteria are met; otherwise, False."""
        if validation_loss < self.min_validation_loss:

            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model_state)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def itr_merge(itrs):
    """
    Merges multiple iterators into a single iterator.

    Args:
        itrs (list): List of iterators to be merged.

    Returns:
        iterator: Merged iterator.
    """
    for itr in itrs:
        for v in itr:
            yield v




def test_patient(loaders, model, maximum_cbg, minimum_cbg):
    """
    Test the model on patient data.

    Args:
        loaders (list): List of data loaders.
        model (torch.nn.Module): PyTorch model.
        maximum_cbg (float): Maximum value used for scaling.
        minimum_cbg (float): Minimum value used for scaling.

    Returns:
        tuple: Mean MSE loss and mean RMSE loss.
    """
    # Define the Mean Squared Error (MSE) loss function
    criterion = nn.MSELoss()
    
    # Lists to store test MSE and RMSE losses
    test_mse_loss = []
    test_rmse_loss = []
    
    # Set the model to evaluation mode
    model.eval()

    # Iterate through batches from merged loaders
    for batch in itr_merge(loaders):
        # Check if all values in dimensions 1 and 2 are NaN
        nan_condition = torch.isnan(batch[0][:, :, 1:3]).all(dim=2).all(dim=1)

        nan_condition_2 = torch.isnan(batch[1]).all(dim=1)

        # Keep only the samples where the condition is False
        inputs = batch[0][~nan_condition]
        real = batch[1][~nan_condition_2]

        # Check if there are any valid samples
        if inputs.shape[0] != 0:
            # Make predictions with the model
            outputs = model(inputs)
            
            # Scale the predictions and real values back to the original range
            outputs = outputs * (maximum_cbg - minimum_cbg) + minimum_cbg
            real = real * (maximum_cbg - minimum_cbg) + minimum_cbg
            
            # Calculate the MSE loss and append to the list
            loss = criterion(outputs, real)
            test_mse_loss.append(loss.item())
            
            # Calculate the RMSE loss and append to the list
            test_rmse_loss.append(torch.sqrt(loss).item())

    # Return the mean test MSE and RMSE losses
    return np.mean(test_mse_loss), np.mean(test_rmse_loss)



def eval_step(loaders, criterion, model):
    """
    Evaluate the model on a validation dataset.

    Args:
        loaders (list): List of data loaders.
        criterion (torch.nn.Module): Loss function.
        model (torch.nn.Module): PyTorch model.

    Returns:
        float: Mean evaluation loss.
    """
    # List to store evaluation losses
    eval_loss = []

    # Set the model to evaluation mode
    model.eval()

    # Iterate through batches from merged loaders
    for inputs, original_values in itr_merge(loaders):
        # Make predictions with the model
        outputs = model(inputs)

        # Calculate the loss using the provided criterion
        loss = criterion(outputs, original_values)
        
        # Append the loss to the list
        eval_loss.append(loss.item())

    # Return the mean evaluation loss
    return np.mean(eval_loss)






def train_step(loaders, criterion, optimizer, model):
    """
    Perform a training step on the model.

    Args:
        loaders (list): List of data loaders.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        model (torch.nn.Module): PyTorch model.

    Returns:
        float: Mean training loss for the epoch.
    """
    # List to store training losses for the epoch
    epoch_loss = []

    # Set the model to training mode
    model.train()

    # Iterate through batches from merged loaders
    for inputs, original_values in itr_merge(loaders):
        # Make predictions with the model
        outputs = model(inputs)
        
        # Calculate the loss using the provided criterion
        loss = criterion(outputs, original_values)
        
        # Append the loss to the list
        epoch_loss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the mean training loss for the epoch
    return np.mean(epoch_loss)
