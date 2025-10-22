import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch.nn as nn
    import torch
    import numpy as np
    import altair as alt
    import polars as pl
    return alt, mo, nn, pl, torch


@app.cell
def _(mo):
    mo.md(r"""#Mean Squared Error""")
    return


@app.cell
def _(nn):
    loss_fun_MSE = nn.MSELoss(reduction='none')
    return (loss_fun_MSE,)


@app.cell
def _(torch):
    y_hat = torch.linspace(-2, 2, 101)
    y = torch.tensor(.5)
    y_hat
    return y, y_hat


@app.cell
def _(loss_fun_MSE, pl, y, y_hat):
    L = pl.DataFrame({
        "row_index": pl.arange(1, 102, eager=True, dtype=pl.Int32),
        "y_hat": y_hat,
        "loss": loss_fun_MSE(y_hat, y.expand_as(y_hat))
    })

    L
    return (L,)


@app.cell
def _(L, alt):
    alt.Chart(data=L).mark_line().encode(x='y_hat', y='loss')
    return


@app.cell
def _(mo):
    mo.md(r"""#Binary Cross-Entropy""")
    return


@app.cell
def _(nn):
    loss_fun_BCE = nn.BCELoss(reduction='none')
    return (loss_fun_BCE,)


@app.cell
def _(loss_fun_BCE, pl, torch):
    y_hat2 = torch.linspace(.001, .999, 101)
    y2a = torch.tensor(0.).expand_as(y_hat2) #stretch the scaler for vectorized operations, expand_as requires a tensor, expand accepts a tuple,
    y2b = torch.tensor(1.).expand_as(y_hat2)
    L2 = pl.DataFrame({
        'y_hat2': y_hat2,
        'y2a_loss': loss_fun_BCE(y_hat2, y2a),
        'y2b_loss': loss_fun_BCE(y_hat2, y2b)
    })
    return (L2,)


@app.cell
def _(L2):
    L2
    return


@app.cell
def _(L2, alt):
    (alt.Chart(L2.unpivot(index='y_hat2'))
        .mark_line()
            .encode(
                x='y_hat2',
                y='value',
                color='variable'
            ))
    return


@app.cell
def _(mo):
    mo.md(r"""#Categorical cross-entropy""")
    return


@app.cell
def _(nn):
    loss_fun_CCE = nn.CrossEntropyLoss()
    return


@app.cell
def _(torch):
    y_hat3 = torch.tensor([[1.,4,3]])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
