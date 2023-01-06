import torch
import torch.nn as nn
import sys

class LSTM2(nn.Module):

    def __init__(self, num_input_features, num_output_features, num_layers=5, num_hidden=12, dtype=torch.float, dropout=0.0):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.num_output_features = num_output_features
        self.num_input_features = num_input_features
        self.num_hidden = num_hidden
        self.lstm = nn.LSTM(num_input_features, num_hidden, num_layers, batch_first=True, dtype=dtype)
        self.fc_1 = nn.Linear(num_layers * num_hidden, 128, dtype=dtype)
        self.fc = nn.Linear(128, num_output_features, dtype=dtype)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # initialize hidden layer and cell to zeros. Given more time, alterative
        # methods for weight initialization could be explored.
        h_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden, dtype=self.dtype)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden, dtype=self.dtype)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # Flatten the last layer of the output
        out = output.contiguous()[:,-self.num_layers:,:].view(-1, self.num_hidden * self.num_layers)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
