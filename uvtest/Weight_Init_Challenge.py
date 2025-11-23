import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F
    from pathlib import Path
    return DataLoader, F, Path, TensorDataset, alt, mo, nn, pl, torch


@app.cell
def _(Path):
    base = Path.cwd()
    mnist = base / 'data' / 'mnist_train_small'
    train_set_path = mnist / 'training_set' / 'training_set_01.parquet'
    test_set_path = mnist / 'test_set' / 'testing_set_01.parquet'
    return test_set_path, train_set_path


@app.cell
def _(pl, test_set_path, train_set_path):
    train_lf = pl.scan_parquet(train_set_path)
    test_lf = pl.scan_parquet(test_set_path)
    return (train_lf,)


@app.cell
def _(DataLoader, TensorDataset, pl, train_lf):
    train_pars_df = train_lf.select(pl.exclude(['id', 'column_1'])).collect().cast(pl.Float32)
    train_labels_ser = train_lf.select(['column_1']).collect().cast(pl.Int64)

    train_set = TensorDataset(train_pars_df.to_torch(), train_labels_ser.to_torch().squeeze(-1))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    train_labels_ser.shape
    return (train_loader,)


@app.cell
def _(F, layer, nn):
    class My_ANN(nn.Module):

        def __init__(self):
            super().__init__()
    
            self.input = nn.Linear(784, 20)
            self.fc1 = nn.Linear(20,10)
            self.output = nn.Linear(10, 10)

            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            return self.output(x)
    return (My_ANN,)


@app.cell
def _(nn, torch, train_loader):
    def training(model, step):
        nn.init.normal_(model.input.weight, mean=0.0, std=step)
        nn.init.normal_(model.fc1.weight, mean=0.0, std=step)
        nn.init.normal_(model.output.weight, mean=0.0, std=step)
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001)
        loss_func = nn.CrossEntropyLoss()
        n_epochs = 100
        epoch_accs = []
        for i in range(n_epochs):
            batch_accs = []
            for X, y in train_loader:
                y_hat = model(X)
                loss = loss_func(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_acc = 100*(torch.argmax(y_hat, dim=1) == y).float().mean().item()
                batch_accs.append(batch_acc)
            epoch_accs.append(sum(batch_accs)/len(batch_accs))
        return epoch_accs
    return (training,)


@app.cell
def _(mo):
    mo.md(r"""
    Challenge: initialize weights to have gaussian distribution from .0001 to 10 (25 logarithmic steps)
    """)
    return


@app.cell
def _():
    #steps = np.logspace(np.log10(.0001),np.log10(10),25)
    return


@app.cell
def _(steps):
    list(steps)
    return


@app.cell
def _(My_ANN):
    my_model = My_ANN()
    return (my_model,)


@app.cell
def _(my_model, training):
    training_acc = training(my_model)
    return


@app.cell
def _(My_ANN, steps, training):
    acc = []
    for step in steps:
        model = My_ANN()
        acc.append(training(model, step))
    return (acc,)


@app.cell
def _(acc):
    acc
    return


@app.cell
def _(acc, pl):
    dfs = []
    for i, epoch_accs in enumerate(acc):
        n_epochs = len(epoch_accs)
        df_tmp = pl.DataFrame({
            "step": [i] * n_epochs,
            "epoch": list(range(n_epochs)),
            "acc": epoch_accs
        })
        dfs.append(df_tmp)

    df = pl.concat(dfs).cast(pl.Float32)
    return (df,)


@app.cell
def _(alt, df):
    alt.Chart(df).mark_line().encode(
        x = 'epoch:Q',
        y = 'acc:Q',
        color = 'step:N'
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(acc):
    acc
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
