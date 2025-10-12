import polars as pl
import httpx
from pathlib import Path
from scipy import stats
import altair as alt


                    #1.1 
def collect_headers(path: Path, separator: str):
    with open(path, 'r') as f:
        #collect first line
        headers = f.readline().strip()
        #list headers
        headers_list = headers.split(separator)
        #remove quotes (polars strips them automagically and performing the operation here is involves less manual coding)
        headers_list = [h.strip('"') for h in headers_list]

    return headers_list


def parquet_path(set_name: str, data_split: str, file_num: int = 1):
    set_folder = Path(set_name)
    split_folder = Path(data_split)
    file_name = f"{data_split}_{file_num:02d}.parquet"
    return Path.cwd()/'data'/set_folder/split_folder/file_name


def split_set_in_parquet(csv_lf: pl.LazyFrame, train_set_path: Path, test_set_path: Path):
    csv_lf.filter((pl.col("id").hash(seed=42) % 10) < 8).sink_parquet(train_set_path)
    csv_lf.filter((pl.col("id").hash(seed=42) % 10) >= 8).sink_parquet(test_set_path)

                    #1 
def prep_local_file(url: str, set_name: str, separator: str = ',') -> Path:
    """Ensure we have a parquet file. Download CSV only if needed."""
    csv_file_path = Path.cwd()/'data'/set_name/f'{set_name}.csv'    # "winequality-red.csv"
    
    parq_train_path = parquet_path(set_name, "Training Set")
    parq_test_path = parquet_path(set_name, "Testing Set")

    if not parq_train_path.exists():

        #download CSV if needed
        if not csv_file_path.exists():
            with httpx.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(csv_file_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)

        # 1.1 readline(first line) and parse headers
        headers = collect_headers(csv_file_path, separator)
        schema_overrides = {header: pl.Float64 for header in headers}        
        #stream to parquet
        csv_lf = pl.scan_csv(
            csv_file_path,
            separator=separator,
            schema_overrides=schema_overrides,
        )
        split_set_in_parquet(csv_lf, parq_train_path, parq_test_path)
    

                    #2
def bin_quality(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("quality").clip(lower_bound=5, upper_bound=6)
    )


                    #3
def rmv_ttlsulf_outliers(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("total sulfur dioxide") < 200)

            
                    #4
def zed_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    #need to benchmark collect_schema()
    feature_columns = [name for name in lf.collect_schema().names() if name != 'quality']

    # pl.col is an expression, to use zscore() the series would first need to be materialized
    normalized_exprs = [
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c) #z-norm
        for c in feature_columns]

    normalized_exprs.append(pl.col("quality").cast(pl.Int8))
    return lf.select(normalized_exprs)


                    #5
def rmv_cloride_abnormal_outlier(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("chlorides") < 10)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
set_name = "winequality-red" #no extension needed (if csv, will convert to parquet, ignore extension)

filepath = prep_local_file(url, set_name, ";")


                    #6
raw_chart = alt.Chart(
    (pl.scan_parquet(filepath)
    .pipe(rmv_ttlsulf_outliers)
    .collect()
    .unpivot()
    )

    ).mark_boxplot(size=60).encode(
        x=alt.X("variable:N", sort=None),
        y="value:Q",
        color="variable:N",

    ).properties(
        width='container',
        height=200
)

                    #7
norm_chart = alt.Chart(
    pl.scan_parquet(filepath)
        .pipe(rmv_ttlsulf_outliers)
        .pipe(bin_quality)
        .pipe(zed_features)
        .pipe(rmv_cloride_abnormal_outlier)
        .collect()
        .unpivot()).mark_boxplot(size=60).encode(

        x=alt.X("variable:N", sort=None),
        y="value:Q",
        color="variable:N",

    ).properties(
    width='container',
    height=200
)

    