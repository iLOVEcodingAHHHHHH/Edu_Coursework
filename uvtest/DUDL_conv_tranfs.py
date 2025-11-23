import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # COURSE: A deep understanding of deep learning
    ## SECTION: Convolution and transformations
    ### LECTURE: Image transformations
    #### TEACHER: Mike X Cohen, sincxpress.com
    ##### COURSE URL: udemy.com/course/deeplearning_x/?couponCode=202401
    """)
    return


@app.cell
def _():
    # import libraries
    import numpy as np
    import torch 

    # NEW!
    import torchvision
    import torchvision.transforms as T

    import matplotlib.pyplot as plt

    return T, np, plt, torch, torchvision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import a dataset
    """)
    return


@app.cell
def _():
    # The list of datasets that come with torchvision: https://pytorch.org/vision/stable/index.html
    return


@app.cell
def _(torchvision):
    # download the CIFAR10 dataset
    cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True)

    print(cdata)
    return (cdata,)


@app.cell
def _(cdata):
    # check out the shape of the dataset
    print( cdata.data.shape )

    # the unique categories
    print( cdata.classes )

    # .targets is a list of targets converted to ints
    print( len(cdata.targets) )
    return


@app.cell
def _(cdata, np, plt):
    # inspect a few random images

    fig,axs = plt.subplots(5,5,figsize=(10,10))

    for ax in axs.flatten():

      # select a random picture
      randidx = np.random.choice(len(cdata.targets))

      # extract that image
      pic = cdata.data[randidx,:,:,:]
      # and its label
      label = cdata.classes[cdata.targets[randidx]]

      # and show!
      ax.imshow(pic)
      ax.text(16,0,label,ha='center',fontweight='bold',color='k',backgroundcolor='y')
      ax.axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Apply some transformations
    """)
    return


@app.cell
def _(T, cdata):
    Ts = T.Compose([ T.ToTensor(),
                     T.Resize(32*4),
                     T.Grayscale(num_output_channels=1)  ])

    # include the transform in the dataset
    cdata.transform = Ts

    # you can also apply the transforms immediately when loading in the data
    # cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True, transform=Ts)


    # Important! Adding a transform doesn't change the image data:
    print(cdata.data[123,:,:,:].shape)
    return (Ts,)


@app.cell
def _(Ts, cdata, plt, torch):
    # apply the transform
    img1 = Ts(cdata.data[123, :, :, :])
    # option 1a: apply the transform "externally" to an image
    img2 = cdata.transform(cdata.data[123, :, :, :])
    fig_1, ax_1 = plt.subplots(1, 3, figsize=(10, 3))
    # option 1b: use the embedded transform
    ax_1[0].imshow(cdata.data[123, :, :, :])
    ax_1[1].imshow(torch.squeeze(img1))
    # let's see what we've done!
    ax_1[2].imshow(torch.squeeze(img2), cmap='gray')
    plt.show()
    return


app._unparsable_cell(
    r"""
    # Note about ToTensor() and normalization:
    ??T.ToTensor()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Additional explorations
    """)
    return


@app.cell
def _():
    # 1) There are many other transforms available in torchvision: https://pytorch.org/vision/stable/transforms.html
    #    Many transformations are useful for data preparation and augmentation. We'll cover some of them later in the course,
    #    but for now, read about RandomCrop(), RandomHorizontalFlip(), and CenterCrop(). Then implement them to understand 
    #    what they do to images.
    #    Tip: It's probably best to test these transforms separately, and on one test image, as we did above.
    #
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
