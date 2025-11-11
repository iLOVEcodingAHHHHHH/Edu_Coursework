import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import altair as alt
    import polars as pl
    import numpy as np
    return mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""
    #gratuisouly complex adding machine (add numbers from -10 to 10)
    """)
    return


@app.cell
def _(pl):
    nums = pl.LazyFrame({'a':pl.int_range(-10,11, eager=True)})
    return


@app.cell
def _(np, pl):
    base_lf = pl.LazyFrame().with_columns(a = pl.int_range(-10,11))
    base_lf = base_lf.join(other=base_lf.rename({'a': 'b'}), how='cross')
    base_lf = base_lf.with_columns(sum = pl.col('a') + pl.col('b'))

    sum_counts_lf = base_lf.group_by('sum').agg(
        pl.len().cast(pl.Float64).alias('pairs')
    )

    count_compliment_lf = sum_counts_lf.with_columns(
        compliment = pl.col('pairs') - pl.col('pairs').max(),
        percent = (pl.col('pairs') / pl.col('pairs').sum()*100)
    
    )

    labels = np.ones((5000, 41))
    labels *= np.arange(-20,21)
    param_1 = labels+ np.random.randint(-10, 11, size=(5000, 41))
    param_2 = labels-param_1
    return count_compliment_lf, labels, param_1, param_2


@app.cell
def _(count_compliment_lf):
    count_compliment_lf.collect()
    return


@app.cell
def _(param_1):
    param_1
    return


@app.cell
def _(param_2):
    param_2
    return


@app.cell
def _(labels):
    labels
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Spent a lot of time toying with my dataset, likely attempting to solve an impossible problem (bias - oversampling edges where I wanted training for 20 as a label vs bias due to variable label training).  Personal take away from project is that some bias may be unavoidable even as simple as the project was.  DUDL instructor randomly calculated sums with oversampling from edge-cases.  Can't justify spending more time on this project.  Further progress can be demonstrated by standard use of lazily evaluated DataFrame.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
