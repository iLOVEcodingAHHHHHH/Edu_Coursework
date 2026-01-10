import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    return np, pl


@app.cell
def _(np):
    n = 10
    a = (np.random.beta(a=1, b=.4, size=n)*100).astype(int)
    b = (np.random.beta(a=1, b=.1, size=n)*100).astype(int)
    return a, b, n


@app.cell
def _(a, b, n, pl):
    deg_freedom = n-1

    a_m = float(a.mean())
    b_m = float(b.mean())

    a_sum = int(a.sum())
    b_sum = int(b.sum())

    dif = a - b

    dif_sq = dif ** 2
    dif_sq_sum = int(dif_sq.sum())

    dif_sum = int((a-b).sum())

    df = (pl.DataFrame({
        'a': a, 'b': b, 'a - b': dif, '(a - b)^2': dif_sq
    }).with_row_index(name='n', offset=1))
    return df, dif_sq_sum, dif_sum


@app.cell
def _(df):
    df
    return


@app.cell
def _(dif_sq_sum, dif_sum, n):
    t = (dif_sum/n)/((dif_sq_sum-(dif_sum**2/n))/((n-1)*n))**.5
    return (t,)


@app.cell
def _(t):
    t
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
