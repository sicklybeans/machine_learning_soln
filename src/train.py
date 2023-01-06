import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import progressbar
import sys
from typing import Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

def train(dataloader, model, optimizer, loss_fn, n_epochs, num_layers, test_dataloader=None):
  loss_tr = np.zeros(n_epochs)
  loss_ts = np.zeros(n_epochs)
  for i in range(n_epochs):
    loss = train_step(i, n_epochs, dataloader, model, optimizer, loss_fn, num_layers)
    loss_tr[i] = loss
    print('Train MSE: %f' % loss)
    if test_dataloader:
      loss = test(test_dataloader, model, loss_fn, num_layers)
      loss_ts[i] = loss
      print('Test  MSE: %f' % loss)
  if test_dataloader:
    return loss_tr, loss_ts
  else:
    return loss_tr

def test(dataloader, model, loss_fn, num_layers):
  model.eval()
  count = 0
  total_loss = 0.0
  with torch.no_grad():
    for i, (x, y, _) in enumerate(dataloader):
      x, y = x.to('cpu'), y.to('cpu')
      pred = model(x)
      total_loss += loss_fn(pred, y[:,-1,:]).item()
      # total_loss += loss_fn(pred, y[:,-num_layers:,:]).item()
      # total_loss += loss_fn(pred, y).item()
      # total_loss += loss_fn(pred[:,-1,:], y[:,-1,:]).item()
      count += len(x)
  return total_loss / count

def train_step(epoch_idx, n_epochs, dataloader, model, optimizer, loss_fn, num_layers):
  model.train()
  count = 0
  total_loss = 0.0
  with get_prog_bar(epoch_idx, n_epochs, len(dataloader)) as bar:
    bar.update(0, epoch_loss=0)
    for i, (x, y, _) in enumerate(dataloader):
      x, y = x.to('cpu'), y.to('cpu')
      optimizer.zero_grad()
      pred = model(x)
      loss = loss_fn(pred, y[:,-1,:])
      # loss = loss_fn(pred[:,-1,:], y[:,-1,:])
      loss.backward()
      optimizer.step()
      count += len(x)
      total_loss += loss.item()
      bar.update(i + 1, epoch_loss=(total_loss / count))
  return total_loss / count

def get_prog_bar(epoch_idx, n_epochs, n_batches):
  widgets = [
    'Epoch %2d / %2d | ' % (epoch_idx + 1, n_epochs),
    progressbar.Percentage(),
    progressbar.Bar(),
    '  ',
    progressbar.ETA(),
    ' | ',
    progressbar.Variable('epoch_loss', precision=3, width=12),
  ]
  return progressbar.ProgressBar(max_value=n_batches, widgets=widgets)
