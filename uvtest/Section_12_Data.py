import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    import polars as pl
    import altair as alt
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split as tt_split
    from scipy import stats

    from timeit import timeit
    return TensorDataset, alt, mo, np, pd, pl, stats, torch, tt_split


@app.cell
def _():
    case_list = ['np', 'pd', 'pl', 'tensor']
    attr_list = ['data', 'shape', 'type', 'dtype']
    return (attr_list,)


@app.cell
def _(attr_list, np):
    mtx_np = np.random.randn(10,10)
    mtx_attr_np = dict(zip(attr_list,[mtx_np, mtx_np.shape, type(mtx_np), mtx_np.dtype])) # dtype (singular) is a np.class
    mtx_attr_np
    return (mtx_np,)


@app.cell
def _(attr_list, mtx_np, pd):
    mtx_pd = pd.DataFrame(mtx_np) #columns arg accepts list for naming
    mtx_attr_pd = dict(zip(attr_list, [mtx_pd.head(1), mtx_pd.shape, type(mtx_pd), mtx_pd.dtypes])) #dtypes is pd.Series
    mtx_attr_pd
    return (mtx_pd,)


@app.cell
def _(attr_list, mtx_np, pl):
    mtx_pl = pl.from_numpy(mtx_np)
    mtx_attr_pl = dict(zip(attr_list, [mtx_pl.head(1), mtx_pl.shape, type(mtx_pl), mtx_pl.dtypes])) #dtypes is py.List
    mtx_attr_pl
    return


@app.cell
def _(mtx_pd, pl):
    mtx_pd_pl = pl.from_pandas(mtx_pd.head(1))
    mtx_pd_pl
    return


@app.cell
def _(np, torch):
    test2 = torch.tensor([[0,1,2,3,4]])
    test = torch.tensor([0,1,2,3,4])
    test3 = torch.tensor(np.random.randn(5,5))
    test4 = test2.T
    test45 = test2 = torch.tensor([[0],[1],[2],[3],[4]])
    test45.shape
    return test, test2, test3, test4


@app.cell
def _(test, test2, test3, test4):
    print(
        test.shape,
        test2.shape,
        test3.shape,
        test4.shape

    )
    return


@app.cell
def _(TensorDataset, test3, test4):
    test_TD = TensorDataset(test3, test4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Data Augmentation: add distance from origin as feature
    """)
    return


@app.cell
def _(np):
    num_cluster = 300
    blur = 1

    A = [1,1]
    B = [5,1]
    C = [4,3]

    a = [A[0]+np.random.randn(num_cluster)*blur, A[1]+np.random.randn(num_cluster)*blur]
    b = [B[0]+np.random.randn(num_cluster)*blur, B[1]+np.random.randn(num_cluster)*blur]
    c = [C[0]+np.random.randn(num_cluster)*blur, C[1]+np.random.randn(num_cluster)*blur]
    return a, b, c


@app.cell
def _(a, b, c, pl):
    lf_a = (
        pl.LazyFrame({"xcor": a[0], "ycor": a[1]})
        .cast(dtypes=pl.Float32)
        .with_columns(pl.lit(0, dtype=pl.Float64).alias("label"))
        .with_columns(hypot = (pl.col("xcor") ** 2 + pl.col("ycor") ** 2)**.5)
    )
    lf_b = (
        pl.LazyFrame({"xcor": b[0], "ycor": b[1]})
        .cast(dtypes=pl.Float32)
        .with_columns(pl.lit(1, dtype=pl.Float64).alias("label"))
        .with_columns(hypot = (pl.col("xcor") ** 2 + pl.col("ycor") ** 2)**.5)
    )
    lf_c = (
        pl.LazyFrame({"xcor": c[0], "ycor": c[1]})
        .cast(dtypes=pl.Float32)
        .with_columns(pl.lit(2, dtype=pl.Float64).alias("label"))
        .with_columns(hypot = (pl.col("xcor") ** 2 + pl.col("ycor") ** 2)**.5)
    )
    df_abc = pl.concat([lf_a, lf_b, lf_c]).collect()
    df_abc = df_abc.select('xcor','ycor','hypot','label')
    df_abc
    return (df_abc,)


@app.cell
def _(alt, df_abc):
    chart = (
        alt.Chart(df_abc)
        .mark_point(size=50)
        .encode(
            x="xcor",
            y="ycor",
            color="label:N",  # Different colors per cluster
            shape="label"   # Different shapes per cluster
        )
    )
    chart
    return


@app.cell
def _(df_abc, pl, tt_split):
    features = df_abc.select(pl.exclude('label')).to_torch()
    labels = df_abc.select('label').to_torch()
    X_tr, X_ts, y_tr, y_ts = tt_split(features, labels, test_size=0.2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #resulting t-test failed - will discontinue exercise.  The model should learn simple relations on it's own.
    """)
    return


@app.cell
def _(stats):
    acc1 = [1,2,3,4,5]
    acc2 = [2,3,4,5,6]
    t,p = stats.ttest_ind(acc1, acc2)
    return p, t


@app.cell
def _(t):
    t
    return


@app.cell
def _(p):
    p # p < 0.5 indicates statistically significance
    return


@app.cell
def _(mo):
    mo.md(r"""
    Save network: `torch.save(<\model>.state_dict(), '<\filename>.pt')
    Load network: `
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
