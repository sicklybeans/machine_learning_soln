import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import progressbar
import sys
import os
import matplotlib.pyplot as plt
from typing import Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.common import make_dir
from src.data import CDataset
from src.models import *


class PathGen():

    def __init__(self, dataset, model, outdir, use_existing=False):
        self.outdir = outdir
        self.dataset = dataset
        self.model = model
        self.ys_file = os.path.join(self.outdir, 'ys.pt')
        self.ts_file = os.path.join(self.outdir, 'ts.pt')
        self.ys_act_file = os.path.join(self.outdir, 'ys_act.pt')
        if use_existing:
            self.load_files()
        if not hasattr(self, 'ts'):
            self.generate_data()

    def generate_plots(self, i1, i2):
        for i in range(self.dataset.get_n_outputs()):
            plt.clf()
            fname = os.path.join(self.outdir, 'y-%d.png' % i)
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            ax.plot(self.ts[i1:i2], self.ys[i1:i2,i], label='Pred data')
            ax.plot(self.ts[i1:i2], self.ys_act[i1:i2,i], label='Real data')
            ax.legend()
            fig.savefig(fname)

    def load_files(self):
        if os.path.exists(self.ts_file):
            self.ts = torch.load(self.ts_file)
            self.ys = torch.load(self.ys_file)
            self.ys_act = torch.load(self.ys_act_file)

    def save_files(self):
        torch.save(self.ts, self.ts_file)
        torch.save(self.ys, self.ys_file)
        torch.save(self.ys_act, self.ys_act_file)

    def generate_data(self):
        self.model.eval()
        with torch.no_grad():
            self.ts = torch.empty(len(ds), dtype=torch.int64)
            self.ys = torch.empty(len(ds), self.dataset.get_n_outputs())
            self.ys_act = torch.empty(len(ds), self.dataset.get_n_outputs())
            for i in progressbar.progressbar(range(len(ds))):
                x, y_act, t = ds[i]
                y = model(x[None,:])
                self.ts[i] = t[-1]
                self.ys[i] = y[0]
                self.ys_act[i] = y_act[-1]
        self.save_files()


ds = CDataset('Test4.csv', 8)
model = LSTM2(ds.get_n_inputs(), ds.get_n_outputs())
model.load_state_dict(torch.load('outputs/ltsm2/model.pt'))
pg = PathGen(ds, model, 'outputs/ltsm2', use_existing=False)
pg.generate_plots(10000, 11000)