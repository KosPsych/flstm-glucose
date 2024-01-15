
import pandas as pd
from dataset import *
from model import *
from constants import *
from data_utils import *
from train_utils import *
import sys
import warnings
import numpy as np
# Filter or suppress warnings
warnings.filterwarnings("ignore")



df_patient1 = pd.read_csv('/home/kospsych/Desktop/projects/glucose/data/test/540-ws-testing_processed.csv')
df_patient2 = pd.read_csv('/home/kospsych/Desktop/projects/glucose/data/test/544-ws-testing_processed.csv')

df_patient1, df_patient2 = preprocess_test(df_patient1, df_patient2)

input_size = 10
output_size = 1


dataset_1 = TimeSeriesDataset(df_patient1, 10)
patient1_loader = DataLoader(dataset_1, batch_size=32, shuffle=False)

dataset_2 = TimeSeriesDataset(df_patient2, 10)
patient2_loader = DataLoader(dataset_2, batch_size=32, shuffle=False)


criterion = nn.MSELoss()
model = TS_MODEL(input_size,  output_size)
model.load_state_dict(torch.load(MODEL_PATH + 'model.pth'))

scaler = joblib.load(SAVE_PATH + 'scaler_model.joblib')

minimum_cbg = scaler.data_min_[0]
maximum_cbg = scaler.data_max_[0]


loss = test_patient(patient1_loader, model, criterion, maximum_cbg, minimum_cbg)
loss_2 = test_patient(patient2_loader, model, criterion, maximum_cbg, minimum_cbg)

print('Loss', (loss+loss_2)/2)
