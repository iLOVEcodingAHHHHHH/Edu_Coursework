import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    alt.data_transformers.disable_max_rows()
    from altscratch import norm_edu_lvl
    return alt, norm_edu_lvl, pl


@app.cell
def _(norm_edu_lvl, pl):
    lf = pl.scan_csv("Salary_Data.csv", infer_schema_length=1000).pipe(norm_edu_lvl)
    return (lf,)


@app.cell
def _(alt, lf):
    source = lf.collect()

    alt.Chart(source).mark_bar().encode(
        x='Education Level',
        y='count():Q',
        color='Education Level'
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
