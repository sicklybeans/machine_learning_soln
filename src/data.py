import itertools
import pandas as pd
import torch
import sys
import numpy as np

# Number of features in label (number of forces)
NUM_OUT = 6

# Number of features in processed input
NUM_INPUT = 2 * (3 + 6)

# Predetermined values to normalize the forces by
FX_RANGE = [-2500, 2500]
FY_RANGE = [-2500, 2500]
FZ_RANGE1 = [-100, 3500]
FZ_RANGE2 = [-2500, 100]

class CDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, sequence_length, transient_time=500):
        super().__init__()
        self.sequence_length = sequence_length
        self.transient_time = transient_time

        dataframe = pd.read_csv(csv_file)
        self.dataframe = dataframe
        length = len(dataframe) - transient_time

        def seq_to_tensor(pd_seq):
            return torch.from_numpy(pd_seq[transient_time:].values).float()

        self._data_y = torch.empty(length, NUM_OUT)
        self._data_x = torch.empty(length, NUM_INPUT)
        self._times = dataframe.t[transient_time:].to_numpy(dtype=np.int64)

        # Generate column names so we can handle different types separately.
        force_cols = ['f%s_%d' % p for p in itertools.product(['x', 'y', 'z'], [1, 2])]
        pos_cols = ['%s_enc_%d' % p for p in itertools.product(['x', 'y', 'z'], [1, 2])]
        ang_cols = ['%s_enc_%d' % p for p in itertools.product(['a', 'b', 'c'], [1, 2])]

        # Load forces into y data
        for i, force_col in enumerate(force_cols):
            self._data_y[:,i] = seq_to_tensor(getattr(dataframe, force_col))

        # Load x/y/z positions into x data
        for i, pos_col in enumerate(pos_cols):
            self._data_x[:,i] = seq_to_tensor(getattr(dataframe, pos_col))

        # For each angle, generate two columns, one for sine and one for cosine.
        for i, ang_col in enumerate(ang_cols):
            self._data_x[:,2*i + 6] = torch.sin(seq_to_tensor(getattr(dataframe, ang_col)))
            self._data_x[:,2*i + 7] = torch.cos(seq_to_tensor(getattr(dataframe, ang_col)))

        self._times = self._times - self._times[0]
        # from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # self._data_x = torch.from_numpy(StandardScaler().fit_transform(self._data_x)).float()
        # self._data_y = torch.from_numpy(MinMaxScaler().fit_transform(self._data_y)).float()
        self._normalize_x_data()
        self._normalize_y_data()

    def _normalize_x_data(self):
        # Normalize all inputs by their standard deviation and mean
        for i in range(NUM_INPUT):
            mean = torch.mean(self._data_x[:,i])
            std = torch.std(self._data_x[:,i])
            self._data_x[:,i] = (self._data_x[:,i] - mean) / std

    def _normalize_y_data(self):
        """
        We can't use the y values to normalize the y data because this is cheating! Instead we should
        rescale the force values by some pre-determined minimum/maximum range that is context specific.
        Here I've chosen fixed values based on the three input files.
        """
        for i in [0, 1]:
            self._data_y[:,i] = (self._data_y[:,i] - FX_RANGE[0]) / (FX_RANGE[1] - FX_RANGE[0])
        for i in [2, 3]:
            self._data_y[:,i] = (self._data_y[:,i] - FY_RANGE[0]) / (FY_RANGE[1] - FY_RANGE[0])
        self._data_y[:,4] = (self._data_y[:,4] - FZ_RANGE1[0]) / (FZ_RANGE1[1] - FZ_RANGE1[0])
        self._data_y[:,5] = (self._data_y[:,5] - FZ_RANGE2[0]) / (FZ_RANGE2[1] - FZ_RANGE2[0])

    def __len__(self):
        return len(self._data_y) - self.sequence_length

    def __getitem__(self, i):
        i2 = i + self.sequence_length
        return self._data_x[i:i2], self._data_y[i:i2], self._times[i:i2]

    def get_n_inputs(self):
        return self._data_x.shape[1]

    def get_n_outputs(self):
        return self._data_y.shape[1]

if __name__ == '__main__':
    cd = CDataset('Test1.csv', 100)
    print(cd._times)
