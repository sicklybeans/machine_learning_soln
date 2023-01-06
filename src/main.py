import progressbar
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common import make_dir
from src.data import CDataset
from src.models import *

SEQ_LENGTH = 8
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 0.005
NAME = 'ltsm2'
OUTPUT_DIR = 'outputs'
DROPOUT = 0.2
TRAIN_FILE = 'Test1.csv'
TEST_FILE = 'Test2.csv'

def go(name=NAME, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, dropout=DROPOUT, lr=LR):
    ds_train = CDataset('Test1.csv', seq_length)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ds_test = CDataset('Test2.csv', seq_length)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    model = LSTM2(ds_train.get_n_inputs(), ds_train.get_n_outputs(), dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('Running training: %s' % name)
    train_losses, test_losses = train(dl_train, model, optimizer, loss_fn, n_epochs, test_dataloader=dl_test)
    save_results(name, model, train_losses, test_losses)

def train(dataloader, model, optimizer, loss_fn, n_epochs, test_dataloader=None):
    loss_tr = np.zeros(n_epochs)
    loss_ts = np.zeros(n_epochs)
    for i in range(n_epochs):
        loss = train_step(i, n_epochs, dataloader, model, optimizer, loss_fn)
        loss_tr[i] = loss
        print('Train MSE: %f' % loss)
        if test_dataloader:
          loss = test(test_dataloader, model, loss_fn)
          loss_ts[i] = loss
          print('Test  MSE: %f' % loss)
    if test_dataloader:
        return loss_tr, loss_ts
    else:
        return loss_tr

def test(dataloader, model, loss_fn):
    model.eval()
    count = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (x, y, _) in enumerate(dataloader):
            x, y = x.to('cpu'), y.to('cpu')
            pred = model(x)
            total_loss += loss_fn(pred, y[:,-1,:]).item()
            count += len(x)
    return total_loss / count

def train_step(epoch_idx, n_epochs, dataloader, model, optimizer, loss_fn):
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

def save_results(name, model, train_losses, test_losses):
    output_dir = os.path.join(OUTPUT_DIR, name)

    # Make plot of MSE vs epoch for train and test data
    fig, ax = plt.subplots()
    n_epochs = len(train_losses)
    x = list(range(1, n_epochs + 1))
    ax.set_xlim(0.5, n_epochs + 0.5)
    ax.plot(x, train_losses, '.', label='Train loss')
    ax.plot(x, test_losses, '.', label='Test loss')
    ax.set_title('MSE vs Epoch for %s' % name)
    ax.legend()

    make_dir(output_dir)
    fig.savefig(os.path.join(output_dir, 'mse-vs-time.png'))

    # Save model state
    model_file = os.path.join(output_dir, 'model.pt')
    torch.save(model.state_dict(), model_file)

if __name__ == '__main__':
  go()