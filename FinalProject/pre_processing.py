import pandas as pd
from evaluator import external_variables


def separated_data_csvs():
    """
    Create seperate csvs data
    :return:
    """
    data_df = pd.read_csv("FinalProject/DATA/USCensus1990.data.txt")
    for column in external_variables:
        data_df[[column]].to_csv(f"FinalProject/DATA/{column}.csv")
    new_df = data_df.drop(columns=["caseid", "dAge", "dHispanic", "iYearwrk", "iSex"])
    new_df.to_csv("FinalProject/DATA/clean_USCensus1990.data.csv")