import pandas as pd
from evaluator import external_variables


def separated_data_csvs():
    """
    Create seperate csvs data
    :return:
    """
    data_df = pd.read_csv("./DATA/USCensus1990.data.txt")
    for column in external_variables:
        data_df[[column]].to_csv(f"./DATA/{column}.csv")
    new_df = data_df.drop(columns=["caseid", "dAge", "dHispanic", "iYearwrk", "iSex"])
    new_df.to_csv("./DATA/clean_USCensus1990.data.csv")