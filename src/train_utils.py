
from itertools import zip_longest
import torch
import numpy as np


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



def test_patient(patient_loader, model, criterion, maximum_cbg, minimum_cbg):

    test_loss = []
    model.eval()
    for batch in patient_loader:


        # Check if all values in dimensions 1 and 2 are NaN
        nan_condition = torch.isnan(batch[0][:, :, 1:3]).all(dim=2).all(dim=1)

        nan_condition_2 = torch.isnan(batch[1]).all(dim=1)

        # Keep only the samples where the condition is False

        inputs = batch[0][~nan_condition]
        real = batch[1][~nan_condition_2]

        if inputs.shape[0] != 0:

            outputs = model(inputs)
            outputs = outputs * (maximum_cbg - minimum_cbg) + minimum_cbg
            real = real * (maximum_cbg - minimum_cbg) + minimum_cbg
            loss = criterion(outputs, real)
            test_loss.append(loss.item())

    return np.mean(test_loss)



def eval_step(patient_1_loader, patient_2_loader, criterion, model):

        # Iterate through both DataLoaders until the longer one is exhausted
        eval_loss = []
        model.eval()
        for batch_patient1, batch_patient2 in zip_longest(patient_1_loader, patient_2_loader, fillvalue=None):

            if batch_patient1 is not None and batch_patient2 is not None:
                 inputs = torch.cat((batch_patient1[0], batch_patient2[0]), dim=0)
                 original_values = torch.cat((batch_patient1[1], batch_patient2[1]), dim=0)
            elif batch_patient1 is not None:
                inputs, original_values = batch_patient1
            elif batch_patient2 is not None:
                inputs, original_values = batch_patient2


            outputs = model(inputs)


            loss = criterion(outputs, original_values)
            eval_loss.append(loss.item())

        return np.mean(eval_loss)



def train_step(patient_1_train_loader, patient_2_train_loader, criterion, optimizer, model):

        # Iterate through both DataLoaders until the longer one is exhausted
        epoch_loss = []
        model.train()
        for batch_patient1, batch_patient2 in zip_longest(patient_1_train_loader, patient_2_train_loader, fillvalue=None):

            if batch_patient1 is not None and batch_patient2 is not None:
                 inputs = torch.cat((batch_patient1[0], batch_patient2[0]), dim=0)
                 original_values = torch.cat((batch_patient1[1], batch_patient2[1]), dim=0)
            elif batch_patient1 is not None:
                inputs, original_values = batch_patient1
            elif batch_patient2 is not None:
                inputs, original_values = batch_patient2
            else:
                continue


            outputs = model(inputs)
            
            loss = criterion(outputs, original_values)
            epoch_loss.append(loss.item())


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.mean(epoch_loss)
