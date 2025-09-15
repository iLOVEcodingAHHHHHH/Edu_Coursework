import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import polars as pl

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    from sklearn.model_selection import train_test_split as ttsplit

    from pathlib import Path
    import duckdb
    DATASET_DB_PATH = (Path(__file__).parent/"resources"/"duck"/"datasets.duckdb").as_posix()
    TABLE_NAME = 'wine'
    return (
        DATASET_DB_PATH,
        DataLoader,
        F,
        TABLE_NAME,
        TensorDataset,
        duckdb,
        mo,
        nn,
        pl,
        torch,
        ttsplit,
    )


@app.cell
def _(DATASET_DB_PATH, TABLE_NAME, duckdb, pl):
    def load_table(table_name: str=TABLE_NAME) -> pl.DataFrame:
        with duckdb.connect(DATASET_DB_PATH) as conn:
            return(conn.sql(f'SELECT * FROM "{table_name}"').pl())   

    def serve_column_descriptions(table_name: str=TABLE_NAME):
        with duckdb.connect(DATASET_DB_PATH) as conn:
            return(conn.sql(f'DESCRIBE SELECT * FROM {TABLE_NAME}').fetchall())

    column_descriptions = serve_column_descriptions()
    return column_descriptions, load_table


@app.cell
def _(column_descriptions, mo):
    batch_wdgt = mo.ui.slider(
        steps=[2,4,8,16,32,64,128,256],
        value=8,
        orientation='vertical',
        show_value=True,
        label='Batch Size'
    )

    label_wdgt = mo.ui.dropdown(
        options=[x[0] for x in column_descriptions],
        label= 'Select Labels'
    )

    layer_wdgt = mo.ui.slider(
        start=0, stop=20,
        value=2,
        orientation='vertical',
        show_value=True,
        label='Hidden Layers'
    )

    neuron_wdgt = mo.ui.slider(
        start=0, stop=20,
        value=3,
        orientation='vertical',
        show_value=True,
        label="""Neurons per Layer"""
    )

    mo.hstack([batch_wdgt, mo.hstack([layer_wdgt, neuron_wdgt, label_wdgt], justify='start')], justify='start', gap=2)
    return (label_wdgt,)


@app.cell
def _(TABLE_NAME, load_table):
    wine_df = load_table(TABLE_NAME)
    wine_df
    wine_df.shape
    return


@app.cell
def _(pl):
    def clean_df(df: pl.DataFrame):
        df = df.filter(pl.col('total sulfur dioxide') < 200)
        df = df.with_columns(pl.col("quality").clip(upper_bound=6, lower_bound=5))
        return df
    return (clean_df,)


@app.cell
def _(TensorDataset, label_wdgt, pl, ttsplit):
    def prep_df(df: pl.DataFrame):

        features = df.select([col for col in df.columns if col != label_wdgt.value]).to_torch
        labels = df.select([label_wdgt.value]).to_torch
        print(features.shape, labels.shape)

        X_train, X_test, y_train, y_test = ttsplit(features, labels, test_size=.2, shuffle=True)

        return (TensorDataset(X_train, y_train), 
                TensorDataset(X_test, y_test),
                X_train.shape[1],
                y_train.max().item() + 1)
    return (prep_df,)


@app.cell
def _(F, nn):
    class NN(nn.Module):
        def __init__(self, df,
                      hidden_layers,
                      neurons,
                      n_features,
                      predictions):

            super().__init__()
            self.dict_layers = nn.ModuleDict()
            self.hidden_layers = hidden_layers

            self.dict_layers['input'] = nn.Linear(n_features, neurons)

            for _ in range(self.hidden_layers):
                self.dict_layers[f'hidden{_}'] = nn.Linear(neurons, neurons)

            self.dict_layers['output'] = nn.Linear(neurons, predictions)

        def forward(self, x):

            x = F.relu(self.dict_layers['input'](x))

            for _ in range(self.hidden_layers):
                x = F.relu(self.dict_layers[f'hidden{_}'])(x)

            x = self.dict_layers['output'](x)

            return x
    return


@app.cell
def _(nn, np, testing_set, torch, train_loader):
    def train_model(model, n_epochs = 500, lr=0.1):

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=model.l2)
        loss_calc = nn.CrossEntropyLoss()


        train_acc = []
        test_acc = []
        for i in range(n_epochs):

            batch_acc = []
            model.train()
            for X, y in train_loader:

                y_hat = model(X)
                loss = loss_calc(y_hat, y)

                if model.l1:
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l1_adj = param.abs().sum() * model.l1
                            loss += l1_adj



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                top_pred = torch.argmax(y_hat, axis=1)
                batch_acc.append(100*torch.mean((top_pred==y).float()).item())


            train_acc.append(np.mean(batch_acc))

            model.eval()
            with torch.no_grad():
                X, y = testing_set.tensors
                y_hat = model(X)
                test_pred = torch.argmax(y_hat, axis=1)
                test_acc.append(100*torch.mean((test_pred==y).float()).item())

        return train_acc, test_acc
    return


@app.cell
def _(DataLoader, TABLE_NAME, batch_size, clean_df, load_table, prep_df):
    training_set, testing_set, n_features, n_predictions = prep_df(clean_df(load_table(TABLE_NAME)))
    train_loader = DataLoader(training_set, batch_size, shuffle=True)
    test_loader = DataLoader(testing_set)
    return testing_set, train_loader


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
