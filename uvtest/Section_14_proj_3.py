import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    lf = pl.scan_parquet('winequality-red.parquet')
    lf = lf.with_columns(pl.col('quality').clip(lower_bound=5,upper_bound=6)).with_row_index(name='idx', offset=1)
    rnd_10_nan = lf.collect().sample(n=10)
    return lf, rnd_10_nan


@app.cell
def _(rnd_10_nan):
    rnd_10_nan
    return


@app.cell
def _(lf, rnd_10_nan):
    non_nan = lf.collect().join(other=rnd_10_nan, how='anti', on='idx')
    return


app._unparsable_cell(
    r"""
    non_nan.join(other=rnd_10_nan.with_columns('residual sugar' = pl.lit(0)), on='idx', how='full')
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
