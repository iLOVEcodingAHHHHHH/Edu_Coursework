import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.signal import convolve2d
    from imageio import imread
    import polars as pl
    import altair as alt
    return alt, convolve2d, np, pl


@app.cell
def _(np):
    imgN = 20
    img = np.random.rand(imgN, imgN)

    kernelN = 5
    Y,X = np.meshgrid(np.linspace(-3,3,kernelN), np.linspace(-3,3,kernelN))
    kernel = np.exp(-(X**2+Y**2)/kernelN)
    return img, imgN, kernel, kernelN


@app.cell
def _(kernel):
    kernel.shape
    return


@app.cell
def _(img, imgN, kernel, kernelN, np, pl):
    kernel_df = pl.DataFrame({
        'x': np.tile(np.arange(start=0, stop=kernelN),reps=kernelN)+.5,
        'y': np.repeat(np.arange(start=0, stop=kernelN), repeats=kernelN)+.5,
        'v': kernel.ravel()
    })
    kernel_df
    img_df = pl.DataFrame({
        'x': np.tile(np.arange(start=0, stop=imgN),reps=imgN)+.5,
        'y': np.repeat(np.arange(start=0, stop=imgN), repeats=imgN)+.5,
        'v': img.ravel()
    })
    img_df
    return img_df, kernel_df


@app.cell
def _(alt, img_df, kernel_df):
    k = alt.Chart(kernel_df).mark_square(size=150, opacity=1).encode(
        x = 'x',
        y = 'y',
        color = alt.Color(
            'v:Q',
            scale=alt.Scale(scheme='magma'),
            legend='v:Q'
        )
    ).properties(
        width=100, height =100
    )
    i = alt.Chart(img_df).mark_square(size=150, opacity=1).encode(
        x = 'x',
        y = 'y',
        color = alt.Color(
            'v:Q',
            scale=alt.Scale(scheme='magma'),
            legend='v:Q'
        )
    ).properties(
        width=300
    )
    i | k
    return


@app.cell
def _(img, imgN, kernel, kernelN, np):
    conv_output = np.zeros((imgN, imgN))
    halfKr = kernelN//2

    for row_i in range(halfKr, imgN-halfKr):
        for col_i in range(halfKr, imgN-halfKr):

            piece = img[row_i-halfKr:row_i+halfKr+1,:]
            piece = piece[:, col_i-halfKr:col_i+halfKr+1]

            dotprod = np.sum(piece*kernel[::-1,::-1])

            conv_output[row_i,col_i] = dotprod
    return (conv_output,)


@app.cell
def _(convolve2d, img, kernel):
    conv_output2 = convolve2d(img, kernel, mode='valid')
    return


@app.cell
def _(conv_output):
    conv_output.shape
    return


@app.cell
def _(conv_output, np, pl):
    conv_hw = conv_output.shape[0]
    conv_df = pl.DataFrame({
        'x': np.tile(np.arange(start=0, stop=conv_hw),reps=conv_hw)+.5,
        'y': np.repeat(np.arange(start=0, stop=conv_hw), repeats=conv_hw)+.5,
        'v': conv_output.ravel()
    })
    conv_df
    return (conv_df,)


@app.cell
def _(alt, conv_df):
    c = alt.Chart(conv_df).mark_square(size=150, opacity=1).encode(
        x = 'x',
        y = 'y',
        color = alt.Color(
            'v:Q',
            scale=alt.Scale(scheme='magma'),
            legend='v:Q'
        )
    ).properties(
        width=300
    )
    c
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
