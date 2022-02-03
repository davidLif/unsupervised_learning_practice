import pandas as pd

external_variables = ["dAge", "dHispanic", "iYearwrk", "iSex"]


def create_csvs():
    data_df = pd.read_csv("FinalProject/DATA/USCensus1990.data.txt")
    for column in external_variables:
        data_df[[column]].to_csv(f"FinalProject/DATA/{column}.csv")
    new_df = data_df.drop(columns=["caseid", "dAge", "dHispanic", "iYearwrk", "iSex"])
    new_df.to_csv("FinalProject/DATA/clean_USCensus1990.data.csv")


def load_data():
    data_df = pd.read_csv("FinalProject/DATA/clean_USCensus1990.data.csv")
    pass

load_data()