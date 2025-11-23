import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    import sklearn.metrics as skm
    from sklearn.model_selection import train_test_split as ttsplit

    from preprocessing import ( # see numbering in preprocessing.py

    prep_local_file, # 1
    bin_quality, # 2
    rmv_ttlsulf_outliers, # 3
    zed_features, # 4
    rmv_chloride_abnormal_outlier, # 5
    plot_chart, # 6

    )


    #this is copied to preproc for charting, probably should add parameters to charts so it's not sloppy
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    filename = "winequality-red" #no extension needed (if csv, will convert to parquet, ignore extension)
    return (
        DataLoader,
        F,
        TensorDataset,
        alt,
        bin_quality,
        copy,
        filename,
        mo,
        nn,
        np,
        pl,
        plot_chart,
        prep_local_file,
        rmv_chloride_abnormal_outlier,
        rmv_ttlsulf_outliers,
        torch,
        url,
    )


@app.cell
def _(filename, prep_local_file, url):
    split_set = prep_local_file(url, filename, ";") #1
    return (split_set,)


@app.cell
def _(pl, split_set):
    train_lf = pl.scan_parquet(split_set[0])
    test_lf = pl.scan_parquet(split_set[1])
    lf = pl.concat([train_lf, test_lf])
    train_lf.collect()
    return lf, test_lf, train_lf


@app.cell
def _(lf):
    lf.collect()
    return


@app.cell
def _(bin_quality, lf, plot_chart, rmv_ttlsulf_outliers):
    raw_lf = (
        lf.drop('id')
        .pipe(bin_quality)
        .pipe(rmv_ttlsulf_outliers)
    )
    plot_chart(raw_lf)  #6
    return


@app.cell
def _(pl, rmv_ttlsulf_outliers, train_lf):
    agg_prep_lf = train_lf.drop(['id', 'quality']).pipe(rmv_ttlsulf_outliers)
    def normalize(lf, agg_prep_lf):
        mean_df, std_df = pl.collect_all([agg_prep_lf.mean(), agg_prep_lf.std()])
        temp = lf.select(['id','quality'])
        lf = lf.select([(pl.col(c) - mean_df[c][0]) / std_df[c][0] for c in mean_df.columns])
        lf = pl.concat([lf, temp],how='horizontal')
        return lf

    #figure out what's going on with lf.select(['id'...]) when it's dropped during definition
    return agg_prep_lf, normalize


@app.cell
def _(
    agg_prep_lf,
    bin_quality,
    normalize,
    rmv_chloride_abnormal_outlier,
    test_lf,
    train_lf,
):
    norm_lf = normalize(train_lf, agg_prep_lf).pipe(bin_quality).pipe(rmv_chloride_abnormal_outlier)
    norm_test_lf = normalize(test_lf, agg_prep_lf).pipe(bin_quality)
    norm_lf.collect()
    return norm_lf, norm_test_lf


@app.cell
def _(norm_lf, norm_test_lf, plot_chart):
    plot_chart(norm_test_lf.drop(['id']))
    norm_lf.collect_schema().len() - 1
    return


@app.cell
def _(DataLoader, TensorDataset, norm_lf, norm_test_lf, pl):
    def loader_prep(preprocced_lf, preprocced_test_lf, batch_size=32):
        X_train, y_train, X_test, y_test = (df.to_torch() for df in pl.collect_all([
                                                                preprocced_lf.drop(['id', 'quality']).cast(pl.Float32),
                                                                preprocced_lf.select(pl.col('quality') - 5).cast(pl.Float32),
                                                                preprocced_test_lf.drop(['id', 'quality']).cast(pl.Float32),
                                                                preprocced_test_lf.select(pl.col('quality') - 5).cast(pl.Float32)
                                                                ]))
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=y_test.shape[0])
        return train_loader, test_loader

    train_load, test_load = loader_prep(norm_lf, norm_test_lf)
    norm_lf.drop(['id', 'quality']).cast(pl.Float32).collect()
    return test_load, train_load


@app.cell
def _(F, nn, norm_lf):
    n_features = norm_lf.collect_schema().len() - 2
    n_predictions = 2
    class MyANN(nn.Module):

        def __init__(self):
            super().__init__()


            self.input = nn.Linear(11,16)

            self.fc1 = nn.Linear(16,32)
            self.fc2 = nn.Linear(32,32)

            self.output = nn.Linear(32,1)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.output(x)
    return (MyANN,)


@app.cell
def _(copy, nn, np, test_load, torch, train_load):
    def train_model(model, n_epochs = 1000, lr=0.1):

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_calc = nn.BCEWithLogitsLoss()


        losses= torch.zeros(n_epochs)
        train_acc = []
        test_acc = []
        top_acc = {'best_acc': 0, 'state': None}
        for i in range(n_epochs):

            model.train()
            batch_acc = []
            batch_loss = []

            for X, y in train_load:

                y_hat = model(X)
                loss = loss_calc(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                batch_acc.append(100*torch.mean(((y_hat>0)==y).float()).item())

            train_acc.append(np.mean(batch_acc))

            losses[i] = np.mean(batch_loss)
            model.eval()
            test_acc_epoch = []
            with torch.no_grad():
                for X, y in test_load:
                    preds = torch.sigmoid(model(X))
                    test_acc_epoch.append(100 * torch.mean(((preds > 0.5) == y).float()).item())
                    if test_acc_epoch[-1] > top_acc['best_acc']:
                        top_acc['best_acc'] = test_acc_epoch[-1]
                        top_acc['state'] = copy.deepcopy(model.state_dict())
                        top_acc['epoch'] = i
            test_acc.append(np.mean(test_acc_epoch))

        return train_acc, test_acc, losses, top_acc
    return (train_model,)


@app.cell
def _(np):
    def smooth(x, k=5):
        return np.convolve(x, np.ones(k)/k, mode='same')
    return


@app.cell
def _(MyANN, train_model):
    my_model = MyANN()
    train_acc, test_acc, loss, best_dict = train_model(my_model, n_epochs=250, lr=0.01)
    return best_dict, my_model, test_acc


@app.cell
def _(test_acc):
    test_acc
    return


@app.cell
def _(alt, pl, test_acc):
    df = pl.DataFrame({
        "epoch": range(1, len(test_acc)+1),
        "accuracy": test_acc
    })

    alt.Chart(df).mark_line(point=True).encode(
        x="epoch",
        y="accuracy"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Future features to consider:
        - async if DL -> convert - preproc multi-file (aiohttp offers significant gains over httpx for heavy traffic, async only)
        - flush() and fsync() to preserve data integrity in event of power outage (resume download functionality?)
    """)
    return


@app.cell
def _(best_dict):
    best_dict['epoch']
    return


@app.cell
def _(best_dict):
    best_dict['state']
    return


@app.cell
def _(test_load):
    dir(test_load)
    return


@app.cell
def _(best_dict, my_model, test_load):
    X, y = next(iter(test_load))
    my_model.load_state_dict(best_dict['state'])
    return (X,)


@app.cell
def _(X, my_model, test_load):
    best_pred = (my_model(X)>0).float()
    lab = (test_load.dataset.tensors[1]>0).float()
    return best_pred, lab


@app.cell
def _(best_pred, lab):
    test = (best_pred == lab).float()
    return (test,)


@app.cell
def _(test, torch):
    torch.mean(test)
    return


@app.cell
def _(best_dict):
    best_dict['best_acc']
    return


@app.cell
def _(best_pred, lab, pl):
    best_df = pl.concat([pl.from_torch(best_pred, schema=['subjective']), pl.from_torch(lab, schema=['objective'])], how='horizontal')
    best_df = best_df.with_columns(
        pl.when((pl.col('subjective') == 1) & (pl.col('objective') == 1))
            .then(pl.lit('TP'))
        .when((pl.col('subjective') == 1) & (pl.col('objective') == 0))
            .then(pl.lit('FP'))
        .when((pl.col('subjective') == 0) & (pl.col('objective') == 1))
            .then(pl.lit('FN'))
            .otherwise(pl.lit('TN'))

        .alias('acc')
    )
    return (best_df,)


@app.cell
def _(best_df):
    best_df['acc']
    return


@app.cell
def _(alt, best_df):
    best_heatmap = alt.Chart(best_df).mark_rect().encode(
        y = alt.Y('subjective:O', sort='descending'),
        x = alt.X('objective:O', sort='descending'),
        color=alt.Color('count(acc)', scale=alt.Scale(scheme='blues'), title='matrix'),
    ).properties(width=300, height=300)
    best_hm_labels = (alt.Chart(best_df)
        .transform_aggregate(
            matrix_ct="count()",
            groupby=['subjective', 'objective', 'acc']
        )
        .transform_calculate(
            label = "datum.acc + ' (' + datum.matrix_ct + ')'"
        )
        .mark_text(baseline="middle")
        .encode(
            y = alt.Y('subjective:O', sort='descending'),
            x = alt.X('objective:O', sort='descending', axis=alt.Axis(orient='top', labelAngle=0)),
            text = 'label:N'
        )
    )
    best_chart = best_heatmap + best_hm_labels
    return (best_chart,)


@app.cell
def _(best_chart):
    best_chart
    return


@app.cell
def _():
    return


@app.cell
def _(alt, best_df):
    base = alt.Chart(best_df).encode(
        alt.X('subjective:O', title='Predicted', scale=alt.Scale(domain=[1, 2])),
        alt.Y('objective:O', title='Actual', scale=alt.Scale(domain=[1, 2]))
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('acc:Q', title='Accuracy', scale=alt.Scale(scheme='blues'))
    ).properties(
        width=200,
        height=200
    ).properties(width=200, height=200)
    return base, heatmap


@app.cell
def _(alt, base, heatmap):
    text = base.mark_text(baseline='middle', fontWeight='bold').encode(
        text=alt.Text('count():Q', format='.0f'),
        color=alt.value('black')
    )

    chart = heatmap + text
    chart
    return


@app.cell
def _(best_df):
    test_group = best_df.group_by(["objective", "subjective"]).count()
    test_group
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
