import pandas as pd 
import gzip
import json
from os.path import exists


def getDF(path, parent_path):
    new_path = parent_path + "Movie_and_TV_5.csv"

    if exists(new_path):
        print("Full dataframe already exists...")
        print("Retrieving full dataframe...")
        df = pd.read_csv(new_path)
    else:
        print("Converting json file to dataframe...")
        df = pd.read_json(path, lines=True)

        print("Saving dataframe as CSV...")
        print(new_path)
        df.to_csv(new_path)

    return df