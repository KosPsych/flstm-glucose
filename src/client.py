import argparse
from collections import OrderedDict
import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
import numpy as np
from utils.data_utils import *
from utils.train_utils import *
from dataset import *
import warnings
from model import *
import torch.optim as optim
from utils.constants import *

# Filter or suppress warnings
warnings.filterwarnings("ignore")

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################


def train(model, node_id):
    # Construct the client data path using DATA_PATH and node_id
    client_data_path = DATA_PATH + 'train/' + str(node_id) + '/'
    
    # Get a list of all files in the client data path
    files = get_all_files_in_folder(client_data_path)

    # Read each CSV file into a DataFrame and append to the dataframes list
    dataframes = []
    for file in files:
        dataframes.append(pd.read_csv(file))

    # Preprocess the training dataframes
    patient_dataframes = preprocess_train(dataframes)

    # Define input and output sizes, batch size, local epochs, and learning rate
    input_size = 10
    output_size = 1
    batch_size = 16
    local_epochs = 2
    learning_rate = 0.001

    # Create train and validation DataLoader objects
    train_loaders, val_loaders = create_train_dataloaders(patient_dataframes, input_size, batch_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Early stopping mechanism
    early_stopping = EarlyStopper()

    # List to store training losses
    train_losses = []

    # Training loop
    for epoch in range(local_epochs):
        # Perform training and evaluation steps
        train_loss = train_step(train_loaders, criterion, optimizer, model)
        eval_loss = eval_step(val_loaders, criterion, model)

        # Print training and evaluation loss for each epoch
        print('LOCAL EPOCH:', epoch, 'Client', node_id, 'Train Loss:', train_loss, 'Eval Loss:', eval_loss)
        
        # Append the training loss to the list
        train_losses.append(train_loss)

        # Early stopping
        if early_stopping.early_stop(eval_loss, model.state_dict()) or epoch == local_epochs - 1:
            # Break the loop if early stopping criteria are met or it's the last epoch
            break

    print('---------------')

    # Return the mean of training losses
    return np.mean(train_losses)





def test(model, node_id):
    # Construct the client data path using DATA_PATH and node_id
    client_data_path = DATA_PATH + 'test/' + str(node_id) + '/'
    
    # Get a list of all files in the client data path
    files = get_all_files_in_folder(client_data_path)

    # Read each CSV file into a DataFrame and append to the dataframes list
    dataframes = []
    for file in files:
        dataframes.append(pd.read_csv(file))

    # Preprocess the test dataframes
    dataframes = preprocess_test(dataframes)

    # Define input and output sizes
    input_size = 5
    output_size = 1

    # Create DataLoader objects for each preprocessed DataFrame
    test_loaders = []
    for i in range(len(dataframes)):
        loader = DataLoader(TimeSeriesDataset(dataframes[i], input_size), batch_size=32, shuffle=False)
        test_loaders.append(loader)

    # Load the scaler model
    scaler = joblib.load(MODEL_PATH + 'scaler_model.joblib')

    # Extract the minimum and maximum values from the scaler
    minimum_cbg = scaler.data_min_[0]
    maximum_cbg = scaler.data_max_[0]

    # Call the test_patient function and return its result
    return test_patient(test_loaders, model, maximum_cbg, minimum_cbg)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
node_id = parser.parse_args().node_id

# Load model and data (simple CNN, CIFAR-10)
model = TS_MODEL(10,  1)




# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, node_id):
        super(FlowerClient, self).__init__()

        # Save the input arguments for later use
        self.node_id = node_id 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train(model, self.node_id)
        return self.get_parameters(config={}), 2, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        mse, rmse = test(model, self.node_id)
        import random
        return  0.0, 2, {"MSE": mse , "RMSE":rmse}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(node_id),
)
