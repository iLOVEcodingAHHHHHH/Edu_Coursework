import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    return F, nn, np, torch


@app.cell
def _(torch):
    import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True,
                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=8, 
                                              shuffle=True,
                                              drop_last=True
                                             )
    images, labels = next(iter(train_loader))
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
    plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show()
    return images, labels, plt, torchvision, train_loader, trainset


@app.cell
def _(train_loader):
    def get_img_attrs():
        img = train_loader.dataset[0][0]
        return (
            img.shape[0],
            img.shape[1:]
        )
    return (get_img_attrs,)


@app.cell
def _(F, get_img_attrs, nn, torch):
    def model_gen(print_toggle=False, maps_1=50, maps_2=25, pool_size=2):

        class Cifar_Model(nn.Module):


            def __init__(self, print_toggle):
                super().__init__()

            
                self.conv_pad = 2
            
                kern_h, kern_w = 3, 3 # height, width
                self.kern_size = (kern_h, kern_w)
            
                stride_v, stride_h = 2, 2  # vertical, horizontal
                self.stride = (stride_v, stride_h)
                        
                self.dropout = nn.Dropout(p=0.05)
                self.print = print_toggle

                img_channels, img_size = get_img_attrs()

            
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
                    dummy = torch.zeros(1, img_channels, *img_size)
                    x = self.conv1(dummy)
                    x = F.max_pool2d(x, pool_size)
                    x = self.conv2(x)
                    x = F.max_pool2d(x, pool_size)
                    flat_size = x.numel()
            
                self.fc1 = nn.Linear(flat_size,100)
                self.fc2 = nn.Linear(100, 10)


        
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

        model = Cifar_Model(print_toggle)

        lossfun = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=.001,
            weight_decay = 1e-4
            )

        return model, lossfun, optimizer
    return (model_gen,)


@app.cell
def _(model_gen, torch, train_loader):
    model, lossfun, optimizer = model_gen(True)

    X,y = next(iter(train_loader))
    yHat = model(X)

    print(yHat.shape)

    loss = lossfun(yHat, torch.squeeze(y))
    print(loss)
    return


@app.cell
def _(model_gen, np, train_loader):
    def model_trainer():

        n_epochs = 20
        model, loss_funct, optimizer = model_gen()

        train_loss = []

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

            train_loss.append(np.mean(batch_loss))

        return train_loss, model
            
    return (model_trainer,)


@app.cell
def _(model_trainer):
    fin_loss, fin_model = model_trainer()
    return (fin_model,)


@app.cell
def _(trainset):
    trainset.class_to_idx
    return


@app.cell
def _(fin_model, images):
    test_preds = fin_model(images)
    return (test_preds,)


@app.cell
def _(test_preds, torch):
    [torch.argmax(prediction) for prediction in test_preds]
    return


@app.cell
def _(labels):
    labels
    return


@app.cell
def _(images, labels, plt, torchvision, trainset):
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
    plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show()
    return


@app.cell
def _(labels, test_preds, torch):
    # Get predicted class indices
    pred_classes = torch.argmax(test_preds, dim=1)  # shape: (batch_size,)

    # Compare with true labels
    correct_mask = (pred_classes == labels)         # tensor of True/False
    num_correct = correct_mask.sum().item()         # number of correct predictions
    accuracy = num_correct / labels.size(0)        # batch accuracy

    print(f'Predicted classes: {pred_classes.tolist()}')
    print(f'True labels:       {labels.tolist()}')
    print(f'Number correct:    {num_correct}/{labels.size(0)}')
    print(f'Batch accuracy:    {accuracy:.2%}')
    return


@app.cell
def _(train_loader):
    len(train_loader.dataset)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
