import torch
import torch.nn as nn


class TS_MODEL(nn.Module):
    def __init__(self, input_size, output_size):
        super(TS_MODEL, self).__init__()
        # First LSTM layer
        self.lstm_1 = nn.LSTM(input_size = input_size, hidden_size = 100, batch_first=True)
        # Attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim  = 100, num_heads=1)
        # Second LSTM layer
        self.lstm_2 = nn.LSTM(100, 100, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(100, output_size)

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_size)
        # First LSTM layer
        lstm_1_out, _ = self.lstm_1(x)
        # Attention layer
        att_out, _ = self.self_attention(lstm_1_out.permute(1, 0, 2), lstm_1_out.permute(1, 0, 2), lstm_1_out.permute(1, 0, 2))
        att_out = att_out.permute(1, 0, 2)
        # Residual connection
        concat_out = lstm_1_out + att_out
        # Second LSTM layer
        lstm_out, _  = self.lstm_2(concat_out)
        # Only take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        # Fully connected layer
        output = self.fc(lstm_out)

        return output




class IDRIS_MODEL(nn.Module):
    def __init__(self, input_size, output_size):
        super(IDRIS_MODEL, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, 50, batch_first=True)

        # Fully connected layer
        self.fc_1 = nn.Linear(50, 30)
        self.fc_2 = nn.Linear(30, 30)
        self.fc_3 = nn.Linear(30, output_size)

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_size)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Only take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layer
        output_fc_1 = self.fc_1(lstm_out)
        output_fc_2 = self.fc_2(output_fc_1)
        output_fc_3 = self.fc_3(output_fc_2)

        return output_fc_3
