import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # COURSE: A deep understanding of deep learning
    ## SECTION: CNN milestone projects
    ### LECTURE: Project 2: CIFAR10 Autoencoder
    #### TEACHER: Mike X Cohen, sincxpress.com
    ##### COURSE URL: udemy.com/course/deeplearning_x/?couponCode=202401
    """)
    return


@app.cell
def _():
    # import libraries
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # for importing data
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline

    return DataLoader, F, T, nn, np, plt, torch, torchvision


@app.cell
def _(torch):
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import the CIFAR dataset
    """)
    return


@app.cell
def _(DataLoader, T, torchvision):
    # transformations
    transform = T.Compose([ T.ToTensor(),
                            T.Normalize([.5,.5,.5],[.5,.5,.5])
                           ])

    # import the data and simultaneously apply the transform
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # transform to dataloaders
    batchsize    = 32
    train_loader = DataLoader(trainset,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(testset, batch_size=256)
    return test_loader, train_loader, trainset


@app.cell
def _(plt, train_loader, trainset):
    # inspect a few random images
    _X, _y = next(iter(train_loader))
    _fig, _axs = plt.subplots(4, 4, figsize=(10, 10))
    for _i, ax in enumerate(_axs.flatten()):
        _pic = _X.data[_i].numpy().transpose((1, 2, 0))
        _pic = _pic / 2 + 0.5
        label = trainset.classes[_y[_i]]
        ax.imshow(_pic)
        ax.text(16, 0, label, ha='center', fontweight='bold', color='k', backgroundcolor='y')  # extract that image (need to transpose it back to 32x32x3)
        ax.axis('off')
    plt.tight_layout()  # undo normalization
    plt.show()  # and its label  # and show!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create the CNN-AE model
    """)
    return


@app.cell
def _(F, nn, torch):
    # create a class for the model
    def makeTheAENet(printtoggle=False):

        class aenet(nn.Module):

            def __init__(self, printtoggle):
                super().__init__()
                self.print = printtoggle  # print toggle
                self.encconv1 = nn.Conv2d(3, 16, 4, padding=1, stride=2)
                self.encconv2 = nn.Conv2d(16, 32, 4, padding=1, stride=2)
                self.latent = nn.Conv2d(32, 64, 4, padding=1, stride=2)  ### -------------- encoding layers -------------- ###
                self.decconv1 = nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2)  # first convolution layer
                self.decconv2 = nn.ConvTranspose2d(32, 16, 4, padding=1, stride=2)  # note: using stride instead of pool to downsample
                self.output = nn.ConvTranspose2d(16, 3, 4, padding=1, stride=2)  # output size: (32+2*1-4)/2 + 1 = 16

            def forward(self, x):  # second convolution layer
                if self.print:
                    print(f'Input: {list(x.shape)}')  # output size: (16+2*1-4)/2 + 1 = 8
                x = F.leaky_relu(self.encconv1(x))
                if self.print:  # third convolution layer (latent code layer)
                    print(f'First encoder block: {list(x.shape)}')
                x = F.leaky_relu(self.encconv2(x))  # output size: (8+2*1-4)/2 + 1 = 4
                if self.print:
                    print(f'Second encoder block: {list(x.shape)}')
                x = F.leaky_relu(self.latent(x))  ### -------------- decoding layers -------------- ###
                if self.print:
                    print(f'Third encoder block: {list(x.shape)}')  # first convolution layer
                x = F.leaky_relu(self.decconv1(x))
                if self.print:
                    print(f'First decoder block: {list(x.shape)}')  # second convolution layer
                x = F.leaky_relu(self.decconv2(x))
                if self.print:
                    print(f'Second decoder block: {list(x.shape)}')  # third convolution layer (output)
                x = F.leaky_relu(self.output(x))
                if self.print:
                    print(f'Decoder output: {list(x.shape)}')
                return x
        net = aenet(printtoggle)
        _lossfun = nn.MSELoss()
        _optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-05)
        return (net, _lossfun, _optimizer)  # first encoder layer  # second encoder layer  # latent layer  # first decoder block  # second decoder block  # third decoder block (output)  # create the model instance  # loss function  # optimizer
    return (makeTheAENet,)


@app.cell
def _(makeTheAENet, train_loader):
    # test the model with one batch
    aenet, _lossfun, _optimizer = makeTheAENet(True)
    _X, _y = next(iter(train_loader))
    _yHat = aenet(_X)
    loss = _lossfun(_yHat, _X)
    print(' ')
    # now compute the loss
    print('Loss:')
    print(loss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create a function that trains the AE model
    """)
    return


@app.cell
def _(device, np, test_loader, torch, train_loader):
    # a function that trains the model
    def function2trainTheAEModel(net, lossfun, optimizer):
        numepochs = 20
        net.to(device)
        trainLoss = torch.zeros(numepochs)  # number of epochs
        testLoss = torch.zeros(numepochs)
        for epochi in range(numepochs):
            net.train()  # send the model to the GPU
            batchLoss = []
            for _X, _y in train_loader:
                _X = _X.to(device)  # initialize losses
                _y = _y.to(device)
                _yHat = net(_X)
                loss = _lossfun(_yHat, _X)
                _optimizer.zero_grad()
                loss.backward()  # loop over epochs
                _optimizer.step()
                batchLoss.append(loss.item())
            trainLoss[epochi] = np.mean(batchLoss)  # loop over training data batches
            net.eval()  # switch to train mode
            batchLoss = []
            for _X, _y in test_loader:
                _X = _X.to(device)
                _y = _y.to(device)
                with torch.no_grad():  # push data to GPU
                    _yHat = net(_X)
                    loss = _lossfun(_yHat, _X)
                batchLoss.append(loss.item())
            testLoss[epochi] = np.mean(batchLoss)  # forward pass and loss
        return (trainLoss, testLoss, net)  # backprop  # loss and accuracy from this batch  # end of batch loop...  # and get average losses and accuracies across the batches  #### test performance (here done in batches!)  # switch to test mode  # push data to GPU  # forward pass and loss  # loss and accuracy from this batch  # end of batch loop...  # and get average losses and accuracies across the batches  # end epochs  # function output
    return (function2trainTheAEModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run the AE model and show the results!
    """)
    return


@app.cell
def _(function2trainTheAEModel, makeTheAENet):
    # ~5 minutes with 15 epochs on GPU
    netAE, lossfun, optimizer = makeTheAENet()
    # create a new model (comment out to re-train)
    trainLossAE, testLossAE, netAE = function2trainTheAEModel(netAE, lossfun, optimizer)
    return netAE, testLossAE, trainLossAE


@app.cell
def _(plt, testLossAE, trainLossAE):
    plt.plot(trainLossAE,'s-',label='AE Train')
    plt.plot(testLossAE,'o-',label='AE Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Model loss (final value: %g)'%testLossAE[-1])
    plt.legend()

    plt.show()
    return


@app.cell
def _(netAE, np, plt, test_loader):
    ## show some random examples
    _X, _y = next(iter(test_loader))
    # get some data
    netAE.cpu()
    netAE.eval()
    # forward pass and loss
    _yHat = netAE(_X)
    _fig, _axs = plt.subplots(2, 10, figsize=(14, 4))  # switch to test mode
    for _i in range(10):
        _pic = _yHat[_i, :, :, :].detach().numpy().transpose((1, 2, 0))
        _pic = _pic / 2 + 0.5
        _axs[0, _i].imshow(_pic)
        _axs[0, _i].set_title(f'[ {np.min(_pic):.2f}, {np.max(_pic):.2f} ]', fontsize=10)
        _axs[0, _i].axis('off')
        _pic = _X[_i, :, :, :].detach().numpy().transpose((1, 2, 0))
        _pic = _pic / 2 + 0.5  # undo normalization
        _axs[1, _i].imshow(_pic)
        _axs[1, _i].set_title(f'[ {np.min(_pic):.2f}, {np.max(_pic):.2f} ]', fontsize=10)
        _axs[1, _i].axis('off')
    plt.show()  # undo normalization
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
