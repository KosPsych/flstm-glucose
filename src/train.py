import pandas as pd
import numpy as np
import os
import sys
from data_utils import *
from train_utils import *
from dataset import *
import warnings
from model import *
import torch.optim as optim
from constants import *

# Filter or suppress warnings
warnings.filterwarnings("ignore")




df_patient1 = pd.read_csv('/home/kospsych/Desktop/projects/glucose/data/train/540-ws-training_processed.csv')
df_patient2 = pd.read_csv('/home/kospsych/Desktop/projects/glucose/data/train/544-ws-training_processed.csv')


df_patient1, df_patient2 = preprocess_train(df_patient1, df_patient2)




input_size = 10
output_size = 1
batch_size = 16
epochs = 20
learning_rate = 0.001


patient_1_train_loader, patient_1_val_loader =  create_train_dataloaders(df_patient1, input_size, batch_size)
patient_2_train_loader, patient_2_val_loader =  create_train_dataloaders(df_patient2, input_size, batch_size)


# Create an instance of the model
model = IDRIS_MODEL(input_size,  output_size)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

early_stopping = EarlyStopper()

for epoch in range(epochs):

    train_loss = train_step(patient_1_train_loader, patient_2_train_loader, criterion, optimizer, model)
    eval_loss = eval_step(patient_1_val_loader, patient_2_val_loader, criterion, model)
    print('EPOCH: ', epoch, ' Train Loss:', train_loss, ' Eval Loss: ', eval_loss)

    # early_stopping
    if early_stopping.early_stop(eval_loss, model.state_dict()) or epoch == epochs - 1:
                torch.save(early_stopping.best_model_state, MODEL_PATH + 'model.pth')
                break
