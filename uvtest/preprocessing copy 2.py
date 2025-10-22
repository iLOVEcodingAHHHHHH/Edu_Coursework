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



def split_set_in_parquet(csv_lf: pl.LazyFrame, train_set_path: Path, test_set_path: Path):
    idx_lf = csv_lf.with_row_index('id')
    idx_lf.filter((pl.col("id").hash(seed=42) % 10) < 8).sink_parquet(train_set_path)
    idx_lf.filter((pl.col("id").hash(seed=42) % 10) >= 8).sink_parquet(test_set_path)


def init_data_paths(set_name: str):
    set_name = Path(set_name).stem
    subfolder = Path.cwd() / "data" / set_name
    train_folder = subfolder / "training_set"
    test_folder = subfolder / "test_set"

    for f in (subfolder, train_folder, test_folder):
        f.mkdir(parents=True, exist_ok=True)

    return {
        'csv': subfolder / f"{set_name}.csv",
        'train_01': train_folder / "training_set_01.parquet",
        'test_01': test_folder / "testing_set_01.parquet"
    }



                    #1 
def prep_local_file(url: str, set_name: str, separator: str = ',') -> Path:
    
    paths = init_data_paths(set_name)

    if not paths['train_01'].exists():

        #download CSV if needed
        if not paths['csv'].exists():
            with httpx.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(paths['csv'], "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)

        # 1.1 readline(first line) and parse headers
        headers = collect_headers(paths['csv'], separator)
        schema_overrides = {header: pl.Float32 for header in headers}        
        #stream to parquet
        csv_lf = pl.scan_csv(
            paths['csv'],
            separator=separator,
            schema_overrides=schema_overrides,
        )
        split_set_in_parquet(csv_lf, paths['train_01'], paths['test_01'])

    return (paths['train_01'], paths['test_01'])
    

                    #2
def bin_quality(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("quality").clip(lower_bound=5, upper_bound=6)
    )


                    #3
def rmv_ttlsulf_outliers(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("total sulfur dioxide") < 200)

            
                    #4
def zed_features(lf: pl.LazyFrame, train_set_mean, train_set_std) -> pl.LazyFrame:
    #need to benchmark collect_schema()
    feature_columns = [name for name in lf.collect_schema().names() if name != 'quality']

    # pl.col is an expression, to use zscore() the series would first need to be materialized
    normalized_exprs = [
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c) #z-norm
        for c in feature_columns]

    normalized_exprs.append(pl.col("quality").cast(pl.Int8))
    return lf.select(normalized_exprs)


                    #5
def rmv_chloride_abnormal_outlier(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("chlorides") < 10)



                    #6
def plot_chart(lf: pl.LazyFrame):
    return alt.Chart(
        lf.collect().unpivot()
    ).mark_boxplot(size=60).encode(
        x=alt.X("variable:N", sort=None),
        y="value:Q",
        color="variable:N",
    ).properties(
        width='container',
        height=200
)

    