import pandas as pd 
import gzip
import json
from tqdm import tqdm
from os.path import exists
from personality_neighbourhood import get_neighbourhood
from sklearn.metrics import mean_squared_error
import numpy as np


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
                yn3 = input("Limit by personality or absolute distribution value? [P/A]: ")
                print("Maximum number of users: ", len(df['reviewerID'].value_counts()))
                if yn3.upper() == "A":
                    valid3 = True
                    # get n most common reviewers
                    n = int(input("Enter a number: "))
                    print("Number of reviewers: ", n)

                    valid4 = False
                    while not valid4:
                        yn4 = input("Stratified or Random? [S/R]: ")
                        if yn4.upper() == "S":
                            valid4 = True
                            chosen = find_chosen(df, df_code)
                            reduced_df = stratified_sampling(n, df, chosen)
                        elif yn4.upper() == "R":
                            valid4 = True
                            frequents = df['reviewerID'].value_counts()[:n].index.tolist()
                            chosen = find_chosen(df, df_code)
                            frequents.append(chosen)
                            reduced_df = df[df['reviewerID'].isin(frequents)]
                        else:
                            print("Invalid input - Please enter an 'S' or an 'A'")

                elif yn3.upper() == "P":
                    valid3 = True
                    chosen = find_chosen(df, df_code)
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
                    reduced2_df.to_csv("reduced.csv")
                    print("total reviews: ", reduced2_df[reduced2_df.columns[0]].count())
                    chosen = find_chosen(reduced2_df, df_code)
                    return reduced2_df, chosen

                elif yn2.upper() == "N":
                    valid2 = True
                    chosen = find_chosen(reduced_df, df_code)
                    print("total reviews: ", reduced_df[reduced_df.columns[0]].count())
                    return reduced_df, chosen

        elif yn.upper() == "N":
            valid = True
            chosen = find_chosen(df, df_code)
            return df, chosen


def stratified_sampling(n, df, chosen):
    print("Conducting Stratified Sampling...")
    k = n/5
    steps = 1/(k-1)
    new_df = pd.DataFrame()

    personalities = pd.read_csv(
        "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")

    saved_df = df.copy()
    df = df.merge(personalities, on="reviewerID")

    new_df = df.copy()[0:0]

    domains = ["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"]

    count = 1
    for domain in domains:
        print(str(count) + "/5")
        count += 1
        target = 0
        for i in range(int(k)):
            row_to_add = df.iloc[(df[domain] - target).abs().argsort()[0]]
            new_df = new_df.append(row_to_add)
            target += steps

    new_df.append(new_df)

    # get ids of chosen_users
    ids = new_df["reviewerID"].unique()
    ids = np.append(ids, chosen)

    print("Chosen Stratified Sample")
    final_sample = saved_df[saved_df['reviewerID'].isin(ids)]

    return final_sample




def find_chosen(df, code):
    # users_ratings = {}
    # grouped = df.groupby(['reviewerID'])
    # for name, group in tqdm(grouped):
    #     group = group
    #     counts = dict(group["overall"].value_counts())
    #     if sum(counts.values()) > 300:
    #         users_ratings[name] = counts
    #
    # for k, v in tqdm(users_ratings.items()):
    #     temp = [0, 0, 0, 0, 0]
    #     for k2, v2 in v.items():
    #         temp[k2-1] = v2/sum(v.values())
    #     users_ratings[k] = temp
    #
    # print(users_ratings)
    # print()
    # print()
    #
    # # calc rmse of each list
    # target = [0.2, 0.2, 0.2, 0.2, 0.2]
    # spread = []
    # for k, v in tqdm(users_ratings.items()):
    #     rms = mean_squared_error(target, v, squared=False)
    #     spread.append([k, rms])
    #
    # print(spread)
    # print(sorted(spread, key=lambda x: x[1], reverse=True))
    # print(len(spread))
    #
    # exit()

    if code.upper() == "K":
        # kindle
        return "A1CIS4LOWYGZGA"
    elif code.upper() == "M":
        # movie
        return "A5TZXWU8AALIC"
    elif code.upper() == "V":
        # video games
        return "A2582KMXLK2P06"
    elif code.upper() == "D":
        # digital music
        return "A3W4D8XOGLWUN5"