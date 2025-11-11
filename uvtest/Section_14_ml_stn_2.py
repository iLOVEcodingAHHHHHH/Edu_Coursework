import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np

    from sklearn.model_selection import train_test_split as ttsplit
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, F, TensorDataset, alt, mo, nn, np, pl, torch, ttsplit


@app.cell
def _(mo):
    mo.md(r"""
    https://chatgpt.com/share/6910b45f-79cc-8007-a5b2-e80c6c852229
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, pl, ttsplit):
    lf = pl.scan_parquet('heart_disease').filter(pl.col('chol')<500).cast(pl.Float32)

    for par in ['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
        lf = lf.with_columns((pl.col(par)-pl.col(par).min())/(pl.col(par).max()-pl.col(par).min()))

    lf_pars = lf.select(pl.exclude('target'))
    lf_labels = lf.select(pl.col('target'))

    X_train, X_test, y_train, y_test = ttsplit(lf_pars.collect().to_torch(), lf_labels.collect().to_torch().ravel(), test_size=.2)

    train_set = TensorDataset(X_train, y_train.unsqueeze(1))
    test_set = TensorDataset(X_test, y_test.unsqueeze(1))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_load = DataLoader(test_set)
    return test_load, train_loader


@app.cell
def _(F, nn):
    class My_Net(nn.Module):

        def __init__(self):
            super().__init__()


            self.input = nn.Linear(13,16)

            self.fc1 = nn.Linear(16,32)
            self.fc2 = nn.Linear(32,32)

            self.output = nn.Linear(32,1)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.output(x)
    return (My_Net,)


@app.cell
def _(nn, np, test_load, torch, train_loader):
    def trainer(model):

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_calc = nn.BCEWithLogitsLoss()

        epochs = 300
        epoch_acc = []
        test_acc = []
        for i in range(epochs):
            model.train()
            batch_acc = []
            for X, y in train_loader:
                y_hat = model(X)
                loss = loss_calc(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_acc.append(100*torch.mean(((y_hat>0)==y).float()).item())
            b_len = len(batch_acc)
            b_sum = sum(batch_acc)
            epoch_acc.append(b_sum/b_len)
            with torch.no_grad():
                b_test_acc=[]
                for X, y in test_load:
                    preds = torch.sigmoid(model(X))
                    b_test_acc.append(100 * torch.mean(((preds > 0.5) == y).float()).item())
                    # if test_acc_epoch[-1] > top_acc['best_acc']:
                    #     top_acc['best_acc'] = test_acc_epoch[-1]
                    #     top_acc['state'] = copy.deepcopy(model.state_dict())
                    #     top_acc['epoch'] = i
                test_acc.append(np.mean(b_test_acc))
        return test_acc


    return (trainer,)


@app.cell
def _(My_Net):
    my_model = My_Net()
    return (my_model,)


@app.cell
def _(my_model, pl, trainer):
    df = pl.DataFrame({'acc': trainer(my_model)}).with_row_index(name='epoch', offset=1)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df):
    alt.Chart(df).mark_line().encode(
        x = 'epoch',
        y = 'acc',
    )
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
