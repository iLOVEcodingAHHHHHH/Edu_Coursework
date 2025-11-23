import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import polars as pl
    import altair as alt
    import numpy as np

    import torch.nn as nn
    from torch.nn import functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from torchsummary import summary

    from sklearn.model_selection import train_test_split
    from pathlib import Path
    return DataLoader, F, Path, TensorDataset, nn, np, pl, summary, torch


@app.cell
def _(Path):
    base_path = Path.cwd() / 'data' / 'mnist_train_small'
    train_path = base_path / 'training_set' / 'training_set_01.parquet'
    test_path = base_path / 'test_set' / 'testing_set_01.parquet'
    return test_path, train_path


@app.cell
def _(pl, test_path, train_path):
    train_lf = pl.scan_parquet(train_path)
    test_lf = pl.scan_parquet(test_path)
    return test_lf, train_lf


@app.cell
def _(np, pl, test_lf, train_lf):
    labels = train_lf.select(pl.col('column_1')).collect().to_series().to_numpy()
    data = train_lf.select(pl.exclude(['id', 'column_1'])).collect().to_numpy()

    data_norm = data/np.max(data)
    data_norm = data_norm.reshape(-1,1,28,28)

    test_lbl = test_lf.select(pl.col('column_1')).collect().to_series().to_numpy()
    test_data = test_lf.select(pl.exclude(['id', 'column_1'])).collect().to_numpy()

    test_data_norm = test_data/np.max(test_data)
    test_data_norm = test_data_norm.reshape(-1,1,28,28)
    return data_norm, labels, test_data_norm, test_lbl


@app.cell
def _(test_data_norm):
    test_data_norm.shape # is tuple, marimo's json viewer breaks display
    return


@app.cell
def _(data_norm, labels, test_data_norm, test_lbl, torch):
    data_T = torch.tensor(data_norm).float()
    labels_T = torch.tensor(labels).long()

    test_data_T = torch.tensor(test_data_norm).float()
    test_lbl_T = torch.tensor(test_lbl).long()
    return data_T, labels_T, test_data_T, test_lbl_T


@app.cell
def _(
    DataLoader,
    TensorDataset,
    data_T,
    labels_T,
    test_data_T,
    test_data_norm,
    test_lbl_T,
):
    train_Tds = TensorDataset(data_T, labels_T)
    train_Tdl = DataLoader(train_Tds,
                           batch_size=32,
                           shuffle=True,
                           drop_last=True)

    test_Tds = TensorDataset(test_data_T, test_lbl_T)
    test_Tdl = DataLoader(test_Tds, batch_size=test_data_norm.shape[0])
    return test_Tdl, train_Tdl


@app.cell
def _(train_Tdl):
    train_Tdl.dataset.tensors[0].shape
    return


@app.cell
def _(F, nn, np, torch):
    def create_mnist_net(printtoggle=False):

        class MNISTnet(nn.Module):

            def __init__(self, printtoggle):
                super().__init__()
                self.kern = 5
                self.stride = 1
                self.pad = 1
                self.pool = 2

                self.conv1 = nn.Conv2d(in_channels =1, out_channels=10,
                                       kernel_size = self.kern,
                                       stride = self.stride,
                                       padding = self.pad)

                self.sizeizzle = np.floor((28+2*self.pad-self.kern)/self.stride) + 1

                self.conv2 = nn.Conv2d(10,20,
                                       kernel_size = self.kern,
                                       stride = self.stride,
                                       padding = self.pad)

                expectSize = np.floor((5+2*0-1)/1)+1
                expectSize = 20*int(expectSize**2) # W * H

                self.fc1 = nn.Linear(expectSize,50)

                self.out = nn.Linear(50, 10)

                self.print = printtoggle


            def forward(self, x):

                if self.print:
                    print(f'Input: {x.shape}')

                x = F.relu(F.max_pool2d(self.conv1(x), self.pool ))

                if self.print:
                    print(f'Layer conv1/pool1: {x.shape}')

                x = F.relu(F.max_pool2d(self.conv2(x), self.pool ))

                if self.print:
                    print(f'Layer conv2/pool2: {x.shape}')

                #reshape for linear

                nUnits = x.shape.numel()/x.shape[0]
                x = x.view(-1,int(nUnits))

                if self.print:
                    print(f'Vectorize: {x.shape}')

                # linear layer
                x = F.relu(self.fc1(x))
                if self.print: print(f'Layer fc1: {x.shape}')
                x = self.out(x)
                if self.print: print(f'Layer out: {x.shape}')

                return x

        net = MNISTnet(printtoggle)

        lossfun = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(net.parameters(), lr=.001)

        return net, lossfun, optimizer
    return (create_mnist_net,)


@app.cell
def _(create_mnist_net):
    net, lossfun, optimizer = create_mnist_net(True)
    return lossfun, net


@app.cell
def _(lossfun, net, train_Tdl):
    X, y = next(iter(train_Tdl))
    yHat = net(X)

    print(' ')
    print(yHat.shape)
    print(y.shape)

    loss = lossfun(yHat, y)
    print(' ')
    print(f'Loss: {loss}')
    return


@app.cell
def _(net, summary):
    summary(net,(1,28,28))
    return


@app.cell
def _(create_mnist_net, np, test_Tdl, torch, train_Tdl):
    def my_trainer():

        epochs = 10

        net, lossfun, optimizer = create_mnist_net()

        losses = torch.zeros(epochs)
        train_acc = []
        test_acc = []

        for i in range(epochs):

            net.train()
            batch_acc = []
            batch_loss = []

            for X, y in train_Tdl:

                yHat = net(X)
                loss = lossfun(yHat,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                matches = torch.argmax(yHat,axis=1) == y
                matches_numer = matches.float()
                accuracy_pct = 100*torch.mean(matches_numer)
                batch_acc.append(accuracy_pct)

            train_acc.append(np.mean(batch_acc))

            losses[i] = np.mean(batch_loss)

            net.eval()
            X,y = next(iter(test_Tdl))
            with torch.no_grad():
                yHat = net(X)

            test_acc.append(100*torch.mean((torch.argmax(yHat, axis=1)==y).float()))

        return train_acc, test_acc, losses, net
    return (my_trainer,)


@app.cell
def _(my_trainer):
    train_acc, test_acc, losses, net_2 = my_trainer()
    return losses, test_acc, train_acc


@app.cell
def _(train_acc):
    train_acc
    return


@app.cell
def _(test_acc):
    test_acc
    return


@app.cell
def _(losses):
    losses
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
