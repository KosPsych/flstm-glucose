import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        self.offset = 0

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get a sequence of data starting from the current index


        input_sequence = self.data[idx:idx + self.sequence_length]





        target = self.data.iloc[idx + self.sequence_length]['cbg']




        if input_sequence['cbg'].isna().any() or math.isnan(target):
            input_sequence = input_sequence.applymap(lambda x: np.nan)
            target = np.nan



        input_sequence = torch.Tensor(input_sequence.values)
        target = torch.Tensor([target])



        return input_sequence, target
