import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import polars as pl
    import altair as alt
    import torch
    import torch.nn as nn
    from torch.utils.data import random_split, TensorDataset, DataLoader
    from torch.nn import functional as F
    return DataLoader, F, TensorDataset, alt, nn, np, pl, random_split, torch


@app.cell
def _(DataLoader, TensorDataset, np, random_split, torch):
    n_per_class = 2000
    img_size = 91

    x = np.linspace(-4,4,img_size)
    X,Y = np.meshgrid(x,x)

    widths = [1.8, 2.4]

    images = torch.zeros(2*n_per_class, 1, img_size, img_size)
    labels = torch.zeros(2*n_per_class)

    for i in range(2*n_per_class):

        ro = 2*np.random.randn(2)
        G = np.exp(-((X-ro[0])**2 +(Y-ro[1])**2)/(2*widths[i%2]**2))
        G = G + np.random.randn(img_size,img_size)/5

        images[i,:,:,:] = torch.Tensor(G).view(1,img_size,img_size)
        labels[i] = i%2

    labels = labels[:,None]
    T_lbls = torch.Tensor(labels).float()
    T_imgs = torch.Tensor(images).float()
    T_imgs.shape

    DS = TensorDataset(T_imgs,T_lbls)
    DS_train, DS_test = random_split(DS, [.8, .2])

    DL_train = DataLoader(DS_train, batch_size=32, drop_last=True)
    DL_test = DataLoader(DS_test, batch_size=len(DS_test))
    return DL_test, DL_train, images, img_size


@app.cell
def _(alt, images, img_size, np, pl):
    raw_df = pl.DataFrame({
        'value': images[0].squeeze().ravel(),
        'x': np.tile(np.arange(img_size),img_size),
        'y': np.repeat(np.arange(img_size),img_size)
    })
    raw_chart = alt.Chart(raw_df).mark_square(size=10).encode(
        x = 'x',
        y = 'y',
        color = alt.Color('value').scale(scheme="inferno")
    ).properties(
        width = 300,
        height = 300
    )
    return raw_chart, raw_df


@app.cell
def _(raw_df):
    raw_df
    return


@app.cell
def _(raw_chart):
    raw_chart
    return


@app.cell
def _(F, img_size, nn, np, torch):

    num_c1_filters = 6
    num_c2_filters = 4

    class GausNet(nn.Module):
        def __init__(self):
            super().__init__()

            kern_height = 3
            kern_width = 3
            conv_pad = 1
            conv_stride = 1

            self.conv1 = nn.Conv2d(
                in_channels = 1,
                out_channels = num_c1_filters,
                kernel_size = (kern_height, kern_width),
                padding = conv_pad
            )
        
            self.c_h1 = np.floor((img_size+conv_pad*2-kern_height)/conv_stride)+1
            self.c_w1 = np.floor((img_size+conv_pad*2-kern_width)/conv_stride)+1
            self.conv2 = nn.Conv2d(
                in_channels = num_c1_filters,
                out_channels = num_c2_filters,
                kernel_size = (kern_height, kern_width),
                padding = conv_pad
            )
        
            self.fc1 = nn.Linear(22*22*num_c2_filters,50)
            self.output = nn.Linear(50,1)


        def forward(self, x):
            c1 = self.conv1(x)
            x = F.avg_pool2d(F.relu(c1),(2,2))
            c2 = self.conv2(x)
            x = F.avg_pool2d(F.relu(c2),(2,2))
            x = self.fc1(torch.flatten(x, start_dim=1))
            return (self.output(x), c1, c2)

    net = GausNet()
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    return lossfun, net, num_c1_filters, num_c2_filters, optimizer


@app.cell
def _(DL_test, DL_train, lossfun, net, np, optimizer, torch):
    def my_trainer():

        epochs = 15

        losses = torch.zeros(epochs)
        train_acc = []
        test_acc = []

        for i in range(epochs):

            net.train()
            batch_acc = []
            batch_loss = []

            for X, y in DL_train:

                yHat = net(X)[0]
                loss = lossfun(yHat,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                predictions = (torch.sigmoid(yHat) > 0.5).float()
                matches = (predictions == y).float()
                accuracy_pct = 100 * torch.mean(matches)
                batch_acc.append(accuracy_pct)

            train_acc.append(np.mean(batch_acc))

            losses[i] = np.mean(batch_loss)

            net.eval()
            X,y = next(iter(DL_test))
            with torch.no_grad():
                yHat = net(X)[0]
                predictions = (torch.sigmoid(yHat) > 0.5).float()
                matches = (predictions == y).float()
                test_acc.append(100 * torch.mean(matches))
            
        return train_acc, test_acc, losses, net
    return (my_trainer,)


@app.cell
def _(my_trainer):
    # Training Cell
    train_acc, test_acc, losses, net_2= my_trainer()
    # Training Cell
    return net_2, test_acc, train_acc


@app.cell
def _(pl, test_acc, train_acc):
    acc_df = pl.concat([

        pl.DataFrame({
            'acc': train_acc,
            'set': 'Train'
        })
            .with_columns(pl.col('acc').cast(pl.Float32))
            .with_row_index(name='epoch', offset=1),

        pl.DataFrame({
            'acc': test_acc,
            'set': 'Test'
        })
            .with_columns(pl.col('acc').cast(pl.Float32))
            .with_row_index(name='epoch', offset=1)
    ])
    return (acc_df,)


@app.cell
def _(acc_df, alt):
    acc_chart = alt.Chart(acc_df).mark_line().encode(
        x = 'epoch:O',
        y = alt.Y('acc', scale=alt.Scale(domain=[90, 100], clamp=True)),
        color = 'set'
    ).properties(
        width = 'container',
        height = 300
    )
    acc_chart
    return


@app.cell
def _(acc_df):
    acc_df
    return


@app.cell
def _(images, net_2):
    num_img_to_review = 5
    conv1, conv2 = net_2(images[0:num_img_to_review])[1:3]
    conv1[0][0].shape
    return conv1, conv2, num_img_to_review


@app.cell
def _(conv1, img_size, np, num_c1_filters, num_img_to_review, pl):
    img_T_dict = {}

    for _img_num in range(num_img_to_review):
        for _filter_num in range(num_c1_filters):
        
            flat_filter = conv1[_img_num][_filter_num].ravel().detach()
        
            filter_df = pl.DataFrame({
                'value': flat_filter,
                'y': np.repeat(np.arange(0,img_size), repeats=img_size),
                'x': np.tile(np.arange(0,img_size), reps=img_size)
            }).with_row_index(name='pixel_idx')
        
            img_T_dict[(_img_num, _filter_num)] = filter_df

    return


@app.cell
def _(conv1, num_c1_filters, num_img_to_review):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    _fig = make_subplots(
        rows=num_c1_filters,
        cols=num_img_to_review,
        subplot_titles=[f'Img {i+1}, Flt {j+1}' for j in range(num_c1_filters) for i in range(num_img_to_review)]
    )

    for _img_num in range(num_img_to_review):
        for _filter_num in range(num_c1_filters):
            _filter_data = conv1[_img_num][_filter_num].detach().numpy()
    
            _fig.add_trace(
                go.Heatmap(z=_filter_data, colorscale='Inferno', showscale=False),
                row=_filter_num+1,
                col=_img_num+1
            )

    _fig.update_xaxes(showticklabels=False)
    _fig.update_yaxes(showticklabels=False)
    _fig.update_layout(height=200*num_c1_filters, width=150*num_img_to_review)
    return go, make_subplots


@app.cell
def _(conv2, go, make_subplots, num_c2_filters, num_img_to_review):
    _fig = make_subplots(
        rows=num_c2_filters,
        cols=num_img_to_review,
        subplot_titles=[f'Img {i+1}, Flt {j+1}' for j in range(num_c2_filters) for i in range(num_img_to_review)]
    )

    for _img_num in range(num_img_to_review):
        for _filter_num in range(num_c2_filters):
            _filter_data = conv2[_img_num][_filter_num].detach().numpy()
    
            _fig.add_trace(
                go.Heatmap(z=_filter_data, colorscale='Inferno', showscale=False),
                row=_filter_num+1,
                col=_img_num+1
            )

    _fig.update_xaxes(showticklabels=False)
    _fig.update_yaxes(showticklabels=False)
    _fig.update_layout(height=200*num_c2_filters, width=150*num_img_to_review)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
