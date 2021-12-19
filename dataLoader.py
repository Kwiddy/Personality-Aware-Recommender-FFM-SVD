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

def reduceDF(df):
    valid = False
    while not valid:
        yn = input("Limit number of users? [Y/N]: ")
        if yn.upper() == "Y":
            valid = True

            # get n most common reviewers
            n = 10
            print("Number of reviewers: ", n)
            frequents = df['reviewerID'].value_counts()[:n].index.tolist()
            reduced_df = df[df['reviewerID'].isin(frequents)]

            valid2 = False
            while not valid2:
                yn2 = input("Restrict number of reviews per user? [Y/N]: ")
                if yn2.upper() == "Y":
                    valid2 = True
                    k = 10
                    print("Reviews per user: ", k)
                    reduced2_df = reduced_df.groupby('reviewerID').head(k).reset_index(drop=True)
                    return reduced2_df

                elif yn2.upper() == "N":
                    valid2 = True
                    return reduced_df

        elif yn.upper() == "N":
            valid = True
            return df