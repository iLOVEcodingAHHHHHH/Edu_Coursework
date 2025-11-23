import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    import polars as pl
    import altair as alt
    from pathlib import Path
    return F, Path, alt, nn, np, pl, plt, torch


@app.cell
def _(Path):
    base_path = Path.cwd()
    mnist_folder_path = base_path/'data'/'mnist_train_small'
    train_parq_path = mnist_folder_path/'training_set'/'training_set_01.parquet'
    test_parq_path = mnist_folder_path/'test_set'/'testing_set_01.parquet'
    return test_parq_path, train_parq_path


@app.cell
def _(pl, test_parq_path, train_parq_path):
    df = pl.concat([pl.scan_parquet(train_parq_path).collect(), pl.scan_parquet(test_parq_path).collect()], how='vertical')
    def zs(df, exclusions):
        df = df.with_columns([
        (pl.col(c) / pl.col(c).max()).fill_nan(0).alias(c)
        for c in df.select(pl.exclude(exclusions)).columns
    ])
        return df
    df = zs(df, ['id', 'column_1'])
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(F, nn, torch):
    class My_ANN(nn.Module):

        def __init__(self):
            super().__init__()

            self.drop = nn.Dropout(p=0.1)

            self.input = nn.Linear(784, 250)
            self.fc1 = nn.Linear(250, 50)
            self.fc2 = nn.Linear(50, 250)
            self.output = nn.Linear(250,784)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.sigmoid(self.output(x))
    return (My_ANN,)


@app.cell
def _(My_ANN):
    model = My_ANN()
    return (model,)


@app.cell
def _(df, nn, pl, torch):
    def train_it(model):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        n_epochs = 10000
        for i in range(n_epochs):
            sample = df.sample(n=32, seed=i).lazy()
            X = sample.select(pl.exclude(['id', 'column_1'])).collect().cast(pl.Float32).to_torch()
            y_hat = model(X)
            loss = loss_func(y_hat, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model(df.sample(n=20, seed = 42).select(pl.exclude(['id', 'column_1'])).cast(pl.Float32).to_torch())
    return (train_it,)


@app.cell
def _(model, train_it):
    test = train_it(model)
    return (test,)


@app.cell
def _(test):
    encoded_arr = test.detach().numpy()
    return (encoded_arr,)


@app.cell
def _(encoded_arr):
    encoded_arr
    return


@app.cell
def _(encoded_arr):
    encoded_arr.shape
    return


@app.cell
def _(encoded_arr):
    reshaped_arr = encoded_arr.reshape((encoded_arr.shape[0],28,28))
    #reshaped_df = pl.concat([pl.DataFrame(reshaped_arr[i]) for i in range(10)])
    #reshaped_df = reshaped_df.with_row_index(name = 'Y')
    #reshaped_df = reshaped_df.with_columns(pl.col('Y')%28+1)
    return (reshaped_arr,)


@app.cell
def _(np, pl, reshaped_arr):
    reshaped_df = (
        pl.DataFrame(
            {
                "image": np.repeat(np.arange(reshaped_arr.shape[0]), 28 * 28),
                "y": np.tile(np.repeat(np.arange(28), 28), reshaped_arr.shape[0]),
                "x": np.tile(np.arange(28), 28 * reshaped_arr.shape[0]),
                "value": reshaped_arr.ravel(),
            }
        )
    )
    return (reshaped_df,)


@app.cell
def _(reshaped_df):
    reshaped_df
    return


@app.cell
def _(alt, reshaped_df):
    chart = alt.Chart(reshaped_df).mark_rect().encode(
        x=alt.X('x:O', title=None),       # pixel column
        y=alt.Y('y:O', title=None),  # reverse y so image is upright
        color=alt.Color(
        'value:Q',
        scale=alt.Scale(
            scheme='greys',   # valid scheme
            reverse=True
        )
    ),  # pixel intensity
        tooltip=['image', 'x', 'y', 'value']
    ).facet(
        column='image:N'  # one column per image
    ).properties(
        width=10,
        height=10
    )

    chart
    return


@app.cell
def _(plt, test):
    fig,axs = plt.subplots(2,5,figsize=(10,3))

    for i in range(5):
      #axs[0,i].imshow(X[i,:].view(28,28).detach() ,cmap='gray')
      axs[1,i].imshow(test[i,:].view(28,28).detach() ,cmap='gray')
      #axs[0,i].set_xticks([]), axs[0,i].set_yticks([])
      axs[1,i].set_xticks([]), axs[1,i].set_yticks([])

    plt.suptitle('Disregard the yikes!!!')
    plt.show()
    return


@app.cell
def _(nn):
    type(nn.Module)
    return


@app.cell
def _(nn):
    dir(nn.Module)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
