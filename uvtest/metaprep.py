import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    import torch
    import torch.nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    from sklearn.model_selection import train_test_split as ttsplit

    from preprocessing import ( # see numbering in preprocessing.py

    prep_local_file, # 1
    bin_quality, # 2
    rmv_ttlsulf_outliers, # 3
    zed_features, # 4
    rmv_cloride_abnormal_outlier, # 5
    raw_chart, # 6
    norm_chart, # 7

    )


    #this is copied to preproc for charting, probably should add parameters to charts so it's not sloppy
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    filename = "winequality-red" #no extension needed (if csv, will convert to parquet, ignore extension)
    return (
        bin_quality,
        filename,
        mo,
        norm_chart,
        pl,
        prep_local_file,
        raw_chart,
        rmv_cloride_abnormal_outlier,
        rmv_ttlsulf_outliers,
        url,
        zed_features,
    )


@app.cell
def _(filename, prep_local_file, url):
    filepath = prep_local_file(url, filename, ";") #1 !!!!!!!!!! this is returning only the training data (needs update to return LazyFrame instead of Path)
    return (filepath,)


@app.cell
def _(raw_chart):
    raw_chart  #6
    return


@app.cell
def _(norm_chart):
    norm_chart #7
    return


@app.cell
def _(
    bin_quality,
    filepath,
    pl,
    rmv_cloride_abnormal_outlier,
    rmv_ttlsulf_outliers,
    zed_features,
):
    lf = (
        pl.scan_parquet(filepath)
            .pipe(rmv_ttlsulf_outliers)
            .pipe(bin_quality)
            .pipe(zed_features)
            .pipe(rmv_cloride_abnormal_outlier)
        )
    lf.collect()
    return (lf,)


@app.cell
def _(lf, pl):
    labels = lf.collect().select(pl.col('quality')).to_torch()
    features = lf.collect().select(pl.exclude('quality')).to_torch()
    return features, labels


@app.cell
def _(features, labels):
    print(labels.shape)
    print(features.shape)
    return


@app.cell
def _(pl):
    def train_test_split_lazy(lf: pl.LazyFrame, train_fraction: float = 0.75
                             ) -> tuple[pl.DataFrame, pl.DataFrame]:

        df = (lf.collect()
            .with_columns(pl.all().shuffle(seed=1))
            .with_row_index()
             )
        #returns eager split
        df_train = df.filter(pl.col("index") < pl.col("index").max() * train_fraction)
        df_test = df.filter(pl.col("index") >= pl.col("index").max() * train_fraction)
        return df_train, df_test
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


@app.cell
def _(mo):
    mo.md(
        r"""
    Future features to consider:
        - async if DL -> convert - preproc multi-file (aiohttp offers significant gains over httpx for heavy traffic, async only)
        - flush() and fsync() to preserve data integrity in event of power outage (resume download functionality?)
    """
    )
    return


if __name__ == "__main__":
    app.run()
