import pandas as pd 
import gzip
import json
from os.path import exists
from personality_neighbourhood import get_neighbourhood


def getDF(path, parent_path, extension):
    new_path = parent_path + extension

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


def reduceDF(df, df_code):
    valid = False
    while not valid:
        yn = input("Limit number of users? [Y/N] (Recommended): ")
        if yn.upper() == "Y":
            valid = True
            valid3 = False
            while not valid3:
                yn3 = input("Limit by personality or absolute value (100)? [P/A]: ")
                if yn3.upper() == "A":
                    valid3 = True
                    # get n most common reviewers
                    n = 100
                    print("Number of reviewers: ", n)
                    frequents = df['reviewerID'].value_counts()[:n].index.tolist()
                    chosen = frequents[0]
                    reduced_df = df[df['reviewerID'].isin(frequents)]
                elif yn3.upper() == "P":
                    valid3 = True
                    print("Reducing by personality...")
                    frequents = df['reviewerID'].value_counts().index.tolist()
                    chosen = frequents[0]
                    neighbours_df = get_neighbourhood(chosen, df_code)
                    neighbours = neighbours_df["reviewerID"].unique()
                    reduced_df = df[df['reviewerID'].isin(neighbours)]

                else:
                    print("Invalid input, please enter a 'P' or an 'A'")

            valid2 = False
            while not valid2:
                yn2 = input("Restrict number of reviews per user? [Y/N]: ")
                if yn2.upper() == "Y":
                    valid2 = True
                    k = 50
                    print("Reviews per user: ", k)
                    reduced2_df = reduced_df.groupby('reviewerID').head(k).reset_index(drop=True)
                    return reduced2_df, chosen

                elif yn2.upper() == "N":
                    valid2 = True
                    return reduced_df, chosen

        elif yn.upper() == "N":
            valid = True
            print(df)
            frequents = df['reviewerID'].value_counts().index.tolist()
            chosen = frequents[0]
            return df, chosen