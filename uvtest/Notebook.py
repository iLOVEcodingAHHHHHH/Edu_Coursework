import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import polars as pl

    import torch
    import torch.nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    from sklearn.model_selection import train_test_split as ttsplit

    from pathlib import Path

    import duckdb
    return (Path,)


@app.cell
def _(Path):
    DATASET_DB_PATH = (Path(__file__).parent/"resources"/"duck"/"datasets.duckdb").as_posix()
    TABLE_NAME = 'wine'
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
