import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import polars as pl
    import torch
    return (np,)


@app.cell
def _(np):
    for i in range(20):
        print(np.random.randint(low=0, high=10))
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
