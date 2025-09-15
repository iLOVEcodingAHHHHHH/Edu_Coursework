from resources import helpers as h_funs
from resources.duck.raw_loader import fetch_data, select_data

def clean_data():
    df = fetch_data()
    print(df.shape)
    print(df)

clean_data()