import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Initialize the TimeSeriesDataset.

        Args:
            data (pandas.DataFrame): Time series data.
            sequence_length (int): Length of the input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.offset = 0

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        Get a single item (input sequence and target) from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Input sequence and target.
        """
        # Get a sequence of data starting from the current index
        input_sequence = self.data[idx:idx + self.sequence_length]

        # Extract the target value for the next time step
        target = self.data.iloc[idx + self.sequence_length]['cbg']

        # Check for missing values in input sequence or target
        if input_sequence['cbg'].isna().any() or math.isnan(target):
            input_sequence = input_sequence.applymap(lambda x: np.nan)
            target = np.nan

        # Convert input sequence and target to PyTorch tensors
        input_sequence = torch.Tensor(input_sequence.values)
        target = torch.Tensor([target])

        return input_sequence, target

