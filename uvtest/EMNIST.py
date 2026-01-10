import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, dataloader
    import torch.nn.functional as F
    import altair as alt
    import numpy as np
    import polars as pl
    import copy
    import torchvision
    from sklearn.model_selection import train_test_split


    cdata = torchvision.datasets.EMNIST(root='emnist',split='letters',download=True)
    return TensorDataset, alt, cdata, np, pl, torch, train_test_split


@app.cell
def _(cdata, torch):
    images = cdata.data.view([124800, 1, 28, 28]).float()
    images /= torch.max(images)
    type(cdata.classes)
    return (images,)


@app.cell
def _(images):
    images.shape
    return


@app.cell
def _(cdata):
    while cdata.classes[0] != "a":
        cdata.classes = cdata.classes[1:]
    return


@app.cell
def _(cdata):
    # review label indexing, done different than instructor
    l_cats = cdata.classes
    l_cats
    return


@app.cell
def _(images):
    images[0][0]
    return


@app.cell
def _(images, np, pl):
    test = pl.DataFrame({
        'y': np.tile(27-np.arange(28), 28),
        'x': np.repeat(np.arange(28), 28),
        'value': images[1][0].flatten(),
    })
    return (test,)


@app.cell
def _(test):
    test
    return


@app.cell
def _(alt, test):
    chart = alt.Chart(test).mark_square(size = 60).encode(
        x = 'x',
        y = 'y',
        color = 'value'
    ).properties(
        height = 300,
        width = 300
    )
    chart
    return


@app.cell
def _(DataLoader, TensorDataset, images, labels, train_test_split):
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=.1)

    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0], shuffle = True, drop_last=True)

    img_size = train_loader.dataset.tensors[0].shape[-1]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
