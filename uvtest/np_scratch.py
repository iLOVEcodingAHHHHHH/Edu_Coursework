import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import polars as pl
    return alt, np, pl


@app.cell
def _(np):
    first = np.arange(1,5)
    second = np.array([6,8,3,2])
    return first, second


@app.cell
def _(first, second):
    summation = first + second
    summation
    return


@app.cell
def _(np):
    np.repeat(np.repeat(5,10),repeats=5,axis=0)
    return


@app.cell
def _(np):
    arr = np.random.rand(784,10).T
    return (arr,)


@app.cell
def _(arr):
    arr
    return


@app.cell
def _(arr):
    arr2 = arr.reshape((28,28,10))
    return (arr2,)


@app.cell
def _(arr2):
    arr2.shape
    return


@app.cell
def _(np):
    np.repeat((1,2), 9)
    return


@app.cell
def _(np, pl):
    x = np.array([1,2,3,4,5])
    z = np.array([2,4,6,8,10])
    w = np.meshgrid(x,z)
    df = pl.DataFrame({'x': w[0].ravel(), 'y': w[1].ravel()})
    w
    return (df,)


@app.cell
def _(alt, df):
    alt.Chart(df).mark_point().encode(
        x = 'x',
        y = 'y'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
