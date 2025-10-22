import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from timeit import timeit
    import altair as alt
    import torch
    from torch import nn as nn
    from torch.utils.data import DataLoader,TensorDataset
    import torch.nn.functional as F
    return DataLoader, F, TensorDataset, alt, nn, np, pl, torch


@app.cell
def _():
    n_per_clus = 300
    blur = .9

    A = [1,1]
    B = [5,1]
    C = [4,4]
    return A, B, C, blur, n_per_clus


@app.cell
def _(A, B, C, alt, blur, n_per_clus, np, pl):
    lf = pl.concat([
        pl.LazyFrame(
            {
                'x': A[0]+np.random.randn(n_per_clus)*blur,
                'y': A[1]+np.random.randn(n_per_clus)*blur,
                'Cluster': 1
            }
        ),
        pl.LazyFrame(
            {
                'x': B[0]+np.random.randn(n_per_clus)*blur,
                'y': B[1]+np.random.randn(n_per_clus)*blur,
                'Cluster': 2
            }
        ),
        pl.LazyFrame(
            {
                'x': C[0]+np.random.randn(n_per_clus)*blur,
                'y': C[1]+np.random.randn(n_per_clus)*blur,
                'Cluster': 0
            }
        )
    ])

    # 2. Plot using color and shape encodings
    chart = (
        alt.Chart(lf.collect())
        .mark_point(size=50)
        .encode(
            x="x",
            y="y",
            color="Cluster:N",  # Different colors per cluster
            shape="Cluster:N"   # Different shapes per cluster
        )
    )
    chart
    return (lf,)


@app.cell
def _(lf):
    lf.collect()
    return


@app.cell
def _(pl):
    def split_lf(whole_lf: pl.LazyFrame) -> (pl.LazyFrame, pl.LazyFrame):
        split = pl.collect_all([

            # train features
        whole_lf
            .select(pl.col(['x','y']))
            .with_row_index('idx')
            .filter((pl.col('idx')
            .hash(seed=42) % 10) < 8)
            .drop('idx')
            .cast(pl.Float32),

            # train labels
        whole_lf
            .select(pl.col('Cluster'))
            .with_row_index('idx')
            .filter((pl.col('idx')
            .hash(seed=42) % 10) < 8)
            .drop('idx')
            .cast(pl.Int64),

            # test features
        whole_lf
            .select(pl.col(['x','y']))
            .with_row_index('idx')
            .filter((pl.col('idx')
            .hash(seed=42) % 10) >= 8)
            .drop('idx')
            .cast(pl.Float32),

            # train labels
        whole_lf
            .select(pl.col('Cluster'))
            .with_row_index('idx')
            .filter((pl.col('idx')
            .hash(seed=42) % 10) >= 8)
            .drop('idx')
            .cast(pl.Int64)
        ])

        return [data.to_torch() for data in split]
    return (split_lf,)


@app.cell
def _(DataLoader, TensorDataset, lf, split_lf):
    train_feat, train_lbl, test_feat, test_lbl = split_lf(lf)

    train_lbl = train_lbl.squeeze(1)
    test_lbl = test_lbl.squeeze(1)


    train_data = TensorDataset(train_feat, train_lbl)
    test_data = TensorDataset(test_feat, test_lbl)

    batch_size = 16

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(test_data,
                             batch_size=test_data.tensors[0].shape[0])

    test_data.tensors[0].shape
    return test_loader, train_loader


@app.cell
def _(F, nn, torch):
    # create a class for the model
    def createTheQwertyNet():

      class QwertyNet(nn.Module):
        def __init__(self):
          super().__init__()

          ### input layer
          self.input = nn.Linear(2,8)

          ### hidden layer
          self.fc1 = nn.Linear(8,8)

          ### output layer
          self.output = nn.Linear(8,3)

        # forward pass
        def forward(self,x):
          x = F.relu( self.input(x) )
          x = F.relu( self.fc1(x) )
          return self.output(x)

      # create the model instance
      net = QwertyNet()

      # loss function
      lossfun = nn.CrossEntropyLoss()

      # optimizer
      optimizer = torch.optim.Adam(net.parameters(),lr=.001, weight_decay=.001)

      return net,lossfun,optimizer
    return (createTheQwertyNet,)


@app.cell
def _(createTheQwertyNet):
    net, loss, opt = createTheQwertyNet()
    return (net,)


@app.cell
def _(net, torch):
    test = torch.rand(10,2)
    net(test)
    return


@app.cell
def _(createTheQwertyNet, net, np, test_loader, torch, train_loader):
    def train_model():

        n_epoch = 20

        netwrk, lossfun, optimizer = createTheQwertyNet()

        losses = torch.zeros(n_epoch)
        train_acc = []
        test_acc = []

        for epoch in range(n_epoch):
            netwrk.train()
            batch_acc = []
            batch_loss = []

            for X, y in train_loader:
                y_hat = netwrk(X)
                loss = lossfun(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                matches = (torch.argmax(y_hat, axis=1) == y).float()
                acc_pct = 100*(torch.mean(matches))
                batch_acc.append(acc_pct)

            train_acc.append(np.mean(batch_acc))
            losses[epoch] = np.mean(batch_loss)
            net.eval()
            X, y = next(iter(test_loader))
            with torch.no_grad():
                y_hat = netwrk(X)

            test_acc.append(100*torch.mean((torch.argmax(y_hat, axis=1)==y).float()))

        return train_acc, test_acc, losses, net            
    return (train_model,)


@app.cell
def _(train_model):
    train_acc, test_acc, losses, netwrk = train_model()
    return test_acc, train_acc


@app.cell
def _(alt, pl, test_acc, train_acc):
    lf_train = pl.LazyFrame({'accuracy': pl.Series(train_acc, dtype=pl.Float32), 'split': 'Training'}).with_row_index('epoch')
    lf_test = pl.LazyFrame({'accuracy': pl.Series(test_acc, dtype=pl.Float32), 'split': 'Testing'}).with_row_index('epoch')
    alt.Chart(pl.concat([lf_train, lf_test]).collect()).mark_line().encode(
        x = 'epoch',
        y = 'accuracy',
        color = 'split'
    )
    return lf_test, lf_train


@app.cell
def _(lf_train):
    lf_train.collect()
    return


@app.cell
def _(lf_test):
    lf_test.collect()
    return


@app.cell
def _(np):
    logspacing = np.logspace(np.log10(.0001), np.log10(.1), 20)
    return (logspacing,)


@app.cell
def _(logspacing):
    logspacing
    return


@app.cell
def _():
    10**.1
    return


@app.cell
def _(np):
    np.log10(.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
