import marimo

__generated_with = "0.18.4"
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
    randint = np.random.randint # Exclusive
    return (
        DataLoader,
        F,
        TensorDataset,
        alt,
        nn,
        np,
        pl,
        randint,
        random_split,
        torch,
    )


@app.cell
def _(DataLoader, TensorDataset, np, randint, random_split, torch):
    nImages = 2000
    imgSize = 30
    
    # Initialize images and labels
    img = np.zeros((nImages, imgSize, imgSize), dtype=np.float32)
    label = np.zeros((nImages, 1), dtype=np.float32)
    
    for i in range(nImages):
        # Add noise
        img[i] = np.random.randn(imgSize, imgSize).astype(np.float32)
        
        # Add a random bar
        i1 = np.random.choice(np.arange(2, 28))  # position 2-27
        i2 = np.random.choice(np.arange(2, 6))   # width 2-5
        
        if np.random.randn() > 0:
            img[i, i1:i1+i2, :] = 1  # horizontal (label 0)
        else:
            img[i, :, i1:i1+i2] = 1  # vertical (label 1)
            label[i] = 1
    
    data_set = TensorDataset(torch.from_numpy(img).unsqueeze(1), torch.from_numpy(label))
    train_set, test_set = random_split(dataset=data_set, lengths=(.8, .2))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    return img, label, test_loader, train_loader


@app.cell
def _(img, np, pl):
    test = img[0]
    test_df = pl.DataFrame(data={
        'x': np.tile(np.arange(len(test[0])), len(test[0]))+.5,
        'y': np.repeat(np.arange(len(test[0]))[::-1], len(test[0]))+.5,
        'values': test.flatten()
    })
    test_df
    return (test_df,)


@app.cell
def _(alt, test_df):
    test_chart = alt.Chart(test_df).mark_square(size = 95).encode(
        x = 'x',
        y = 'y',
        color = alt.Color('values', scale=alt.Scale(scheme='magma'))
    ).properties(
        height = 300,
        width = 300
    )
    test_chart
    return


@app.cell
def _(label):
    label
    return


@app.cell
def _(F, nn, torch, train_loader):
    def get_img_attrs():
        img = train_loader.dataset[0][0]
        return (
            img.shape[0:] # 0 is channels for non-greyscale (return multiple slices if channels are needed)
        )
    get_img_attrs()

    def model_gen(print_toggle=False, maps_1=5, maps_2=10, pool_size=2):

        class Psyc_Model(nn.Module):


            def __init__(self, print_toggle):
                super().__init__()


                self.conv_pad = 1

                kern_h, kern_w = 3, 3 # height, width
                self.kern_size = (kern_h, kern_w)

                stride_v, stride_h = 1, 1  # vertical, horizontal
                self.stride = (stride_v, stride_h)

                self.dropout = nn.Dropout(p=0.05)
                self.print = print_toggle

                img_size = get_img_attrs()
                img_channels = 1

                self.conv1 = nn.Conv2d(
                    in_channels=img_channels,
                    out_channels=maps_1,
                    kernel_size=self.kern_size,
                    padding=self.conv_pad,
                    stride=self.stride
                )
                self.bnorm1 = nn.BatchNorm2d(maps_1)


                self.conv2 = nn.Conv2d(
                    in_channels = maps_1,
                    out_channels = maps_2,
                    kernel_size = self.kern_size,
                    padding = self.conv_pad,
                    stride = self.stride
                )
                self.bnorm2 = nn.BatchNorm2d(maps_2)

                with torch.no_grad():
                    dummy = torch.zeros(1, *img_size)
                    x = self.conv1(dummy)
                    x = F.max_pool2d(x, pool_size)
                    x = self.conv2(x)
                    x = F.max_pool2d(x, pool_size)
                    flat_size = x.numel()

                self.fc1 = nn.Linear(flat_size,15)
                self.fc2 = nn.Linear(15, 1)



            def forward(self, x):

                if self.print:
                    print(f'Input: {list(x.shape)}')

                x = self.conv1(x)
                x = self.bnorm1(x)
                x = F.relu(x)
                x = F.max_pool2d(x, pool_size)
                if self.print:
                    print(f'First CPR block: {list(x.shape)}')

                x = self.conv2(x)
                x = self.bnorm2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, pool_size)
                if self.print:
                    print(f'Second CPR block {list(x.shape)}')

                x = x.flatten(start_dim=1)
                if self.print:
                    print(f'Vectorized: {list(x.shape)}')

                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                if self.print:
                    print(f'Final output: {list(x.shape)}')

                return x

        model = Psyc_Model(print_toggle)

        lossfun = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=.001,
            weight_decay = 1e-4
            )

        return model, lossfun, optimizer
    return (model_gen,)


@app.cell
def _(model_gen, train_loader):
    model, lossfun, optimizer = model_gen(True)

    X,y = next(iter(train_loader))
    yHat = model(X)

    print(yHat.shape)

    loss = lossfun(yHat, y)
    print(loss)
    return


@app.cell
def _(model_gen, np, test_loader, torch, train_loader):
    def model_trainer():

        n_epochs = 10
        model, loss_funct, optimizer = model_gen()

        train_loss = []
        test_acc = []

        for epoch in range(n_epochs):

            model.train()
            batch_loss = []

            for X, y in train_loader:

                y_Hat = model(X)
                loss = loss_funct(y_Hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            model.eval()
            with torch.no_grad():
                X, y = next(iter(test_loader))
                y_Hat = model(X)                  # logits
                probs = torch.sigmoid(y_Hat)               # convert to probabilities
                preds = (probs > 0.5).float()            # threshold at 0.5
                accuracy = (preds == y).float().mean().item()

            train_loss.append(np.mean(batch_loss))
            test_acc.append(accuracy)

        return train_loss, test_acc, model
    return (model_trainer,)


@app.cell
def _(model_trainer):
    fin_loss, fin_acc, fin_model = model_trainer()
    return fin_acc, fin_model


@app.cell
def _(fin_acc):
    fin_acc
    return


@app.cell
def _(test_loader):
    next(iter(test_loader))
    return


@app.cell
def _(fin_model, test_loader):
    fin_model(next(iter(test_loader))[0])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
