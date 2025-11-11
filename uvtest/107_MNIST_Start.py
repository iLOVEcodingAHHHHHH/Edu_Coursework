import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    from preprocessing import split_set_in_parquet, init_data_paths
    import altair as alt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    return (
        DataLoader,
        F,
        Path,
        TensorDataset,
        alt,
        init_data_paths,
        nn,
        np,
        pl,
        split_set_in_parquet,
        torch,
    )


@app.cell
def _(Path, pl):
    filename = 'mnist_train_small'
    csv_path = Path().cwd()/f'{filename}.csv'
    csv_lf = pl.scan_csv(csv_path, has_header=False)
    csv_lf.collect()
    return csv_lf, filename


@app.cell
def _(filename, init_data_paths):
    file_paths = init_data_paths(filename)
    file_paths
    return (file_paths,)


@app.cell
def _(csv_lf, file_paths, pl, split_set_in_parquet):
    split_set_in_parquet(csv_lf=csv_lf, train_set_path=file_paths['train_01'], test_set_path=file_paths['test_01'])

    train_lf = pl.scan_parquet(file_paths['train_01'])
    train_labels = train_lf.select('column_1').cast(pl.Int64)
    train_features = train_lf.select(pl.exclude(['id', 'column_1'])
                                     / 255 # normilization lines separated for challenge, section 109 
                                    ).cast(pl.Float32)


    test_lf = pl.scan_parquet(file_paths['test_01'])
    test_labels = test_lf.select('column_1').cast(pl.Int64)
    test_features = test_lf.select(pl.exclude(['id', 'column_1'])
                                   / 255 
                                  ).cast(pl.Float32)
    test_features.collect_schema().len()
    return test_features, test_labels, train_features, train_labels


@app.cell
def _(
    DataLoader,
    TensorDataset,
    test_features,
    test_labels,
    train_features,
    train_labels,
):
    train_set = TensorDataset(train_features.collect().to_torch(), train_labels.collect().to_torch().squeeze(1))
    test_set = TensorDataset(test_features.collect().to_torch(), test_labels.collect().to_torch().squeeze(1))

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

    test_set.tensors[1].shape
    return test_loader, train_loader


@app.cell
def _(F, nn):
    class Mnist_Ann(nn.Module):

        def __init__(self):
            super().__init__()

            self.input = nn.Linear(784, 64)
            self.fc1 = nn.Linear(64,64)
            self.fc2 = nn.Linear(64,32)
            self.output = nn.Linear(32,10)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.output(x)
    return (Mnist_Ann,)


@app.cell
def _(Mnist_Ann, np, test_loader, torch, train_loader):
    def train(model: Mnist_Ann):

        n_epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        batch_acc = []
        batch_loss = []
        test_acc = []
        train_acc = []
        train_loss = []
        test_loss = []

        for epoch in range(n_epochs):

            for X, y in train_loader:
                y_hat = model(X)
                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                pred = torch.argmax(y_hat, axis=1)
                matches = pred == y
                f32_match = matches.float()
                acc = 100*torch.mean(f32_match)
                batch_acc.append(acc)


            train_acc.append(np.mean(batch_acc))
            train_loss.append(np.mean(batch_loss))

            X, y = next(iter(test_loader))
            y_hat = model(X)
            t_loss = criterion(y_hat, y)
            test_loss.append(t_loss.item())
            test_acc.append(100*torch.mean((torch.argmax(y_hat, axis=1)==y).float()))


        return train_acc, test_acc, train_loss, test_loss, batch_loss




    return (train,)


@app.cell
def _(Mnist_Ann, train):
    my_model = Mnist_Ann()
    train_acc, test_acc, train_loss, test_loss, batch_loss = train(my_model)
    return test_acc, test_loss, train_acc, train_loss


@app.cell
def _(pl, test_acc, train_acc):
    train_df = pl.DataFrame(data={'acc': train_acc}).with_row_index(name='epoch', offset=1)
    test_df = pl.DataFrame(data={'acc': test_acc}).with_row_index(name='epoch', offset=1)
    return test_df, train_df


@app.cell
def _(test_df):
    test_df
    return


@app.cell
def _(alt, test_df, train_df):
    acc_ch1 = alt.Chart(train_df).mark_line().encode(
        x='epoch',
        y='acc'
    )
    acc_ch2 = alt.Chart(test_df).mark_line().encode(
        x='epoch',
        y='acc'
    )
    acc_ch1 | acc_ch2
    return


@app.cell
def _(pl, test_loss, train_loss):
    test_lss_df = pl.DataFrame({'Loss': test_loss}).with_row_index(name='epoch', offset=1)
    train_lss_df = pl.DataFrame({'Loss': train_loss}).with_row_index(name='epoch', offset=1)
    return test_lss_df, train_lss_df


@app.cell
def _(alt, test_lss_df, train_lss_df):
    lss_ch1 = alt.Chart(train_lss_df).mark_line().encode(
        y='Loss',
        x='epoch'
    )
    lss_ch2 = alt.Chart(test_lss_df).mark_line().encode(
        y='Loss',
        x='epoch'
    )
    lss_ch1 | lss_ch2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
