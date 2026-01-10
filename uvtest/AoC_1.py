import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import httpx
    import polars as pl
    return (httpx,)


@app.cell
def _():
    url = "https://adventofcode.com/2025/day/1/input"
    return (url,)


@app.cell
def _(httpx, url):
    html = httpx.get(url).text
    return (html,)


@app.cell
def _(html):
    html
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
