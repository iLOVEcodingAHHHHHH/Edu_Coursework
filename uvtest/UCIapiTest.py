import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    return (fetch_ucirepo,)


@app.cell
def _(fetch_ucirepo):
    wine = fetch_ucirepo(id=109)
    return (wine,)


@app.cell
def _(wine):
    wine
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
