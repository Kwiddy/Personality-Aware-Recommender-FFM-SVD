import pandas as pd 
import gzip
import json
from os.path import exists


def getDF(path):
    new_path = [char for char in path]
    i = -1
    while new_path[i] != "/":
        del new_path[i]
    new_path = ''.join(new_path)
    new_path += "Movie_and_TV_5.csv"

    if exists(new_path):
        print("Dataframe already exists...")
        df = pd.read_csv(new_path)
    else:
        print("Converting json file to dataframe...")
        df = pd.read_json(path, lines=True)

        print("Saving dataframe as CSV...")
        print(new_path)
        df.to_csv(new_path)

        print("Printing dataframe...")

    print(df)