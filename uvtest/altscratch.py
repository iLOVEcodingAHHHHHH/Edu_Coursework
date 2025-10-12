import polars as pl

def norm_edu_lvl(lf: pl.LazyFrame) -> pl.LazyFrame:
    educlev_labels = {
    "High School": "HS",
    "Bachelor's": "BS",
    "Bachelor's Degree": "BS",
    "Master's": "MS",
    "Master's Degree": "MS",
    "PhD": "PhD",
    "phD": "PhD"
}
    return lf.with_columns(
        pl.col('Education Level').replace_strict(educlev_labels, default="other")
    )
    