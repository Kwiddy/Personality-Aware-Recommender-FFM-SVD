import pandas as pd 
import gzip
import json
from tqdm import tqdm
from os.path import exists
from personality_neighbourhood import get_neighbourhood
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle


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


def reduceDF(df, df_code, chosen, restrict_reviews, limit_method, limit, sub_limit_method):
    valid = False
    while not valid:
        if limit is not None:
            yn = limit
        else:
            yn = input("Limit number of users? [Y/N] (Recommended): ")
        if yn.upper() == "Y":
            valid = True
            limit = yn
            valid3 = False
            while not valid3:
                if limit_method is not None:
                    yn3 = limit_method
                else:
                    yn3 = input("Limit by personality or absolute distribution value? [P/A]: ")
                if yn3.upper() == "A":
                    print("Maximum number of users: ", len(df['reviewerID'].value_counts()))
                    valid3 = True
                    limit_method = yn3
                    # get n most common reviewers
                    n = int(input("Enter a number: "))
                    print("Number of reviewers: ", n)

                    valid4 = False
                    while not valid4:
                        if sub_limit_method is not None:
                            yn4 = sub_limit_method
                        else:
                            yn4 = input("Stratified or Random? [S/R]: ")
                        if yn4.upper() == "S":
                            sub_limit_method = yn4
                            valid4 = True
                            reduced_df = stratified_sampling(n, df, chosen)
                        elif yn4.upper() == "R":
                            sub_limit_method = yn4
                            valid4 = True
                            frequents = df['reviewerID'].value_counts()[:n].index.tolist()
                            frequents.append(chosen)
                            reduced_df = df[df['reviewerID'].isin(frequents)]
                        else:
                            print("Invalid input - Please enter an 'S' or an 'A'")

                elif yn3.upper() == "P":
                    print("Maximum number of users: ", len(df['reviewerID'].value_counts()))
                    valid3 = True
                    limit_method = yn3
                    valid5 = False
                    while not valid5:
                        if sub_limit_method is not None:
                            yn5 = sub_limit_method
                        else:
                            print("[N] Neighbourhood")
                            # print("[L] Linear Stratified")
                            print("[S] Logarithmic Stratified")
                            yn5 = input("Please select an option above: ")
                        if yn5.upper() == "S":
                            sub_limit_method = yn5
                            valid5 = True
                            stratified = True
                            neighbours_df = get_neighbourhood(chosen, df_code, stratified)
                            neighbours = neighbours_df["reviewerID"].unique()
                            reduced_df = df[df['reviewerID'].isin(neighbours)]
                        elif yn5.upper() == "N":
                            sub_limit_method = yn5
                            valid5 = True
                            stratified = False
                            neighbours_df = get_neighbourhood(chosen, df_code, stratified)
                            neighbours = neighbours_df["reviewerID"].unique()
                            reduced_df = df[df['reviewerID'].isin(neighbours)]
                        else:
                            print("Invalid input")
                else:
                    print("Invalid input, please enter a 'P' or an 'A'")

            valid2 = False
            while not valid2:
                if restrict_reviews is not None:
                    yn2 = restrict_reviews
                else:
                    yn2 = input("Restrict number of reviews per user? [Y/N]: ")
                if yn2.upper() == "Y":
                    restrict_reviews = yn2
                    valid2 = True
                    k = 50
                    print("Reviews per user: ", k)
                    reduced2_df = reduced_df.groupby('reviewerID').head(k).reset_index(drop=True)
                    reduced2_df.to_csv("reduced.csv")
                    print("total reviews: ", reduced2_df[reduced2_df.columns[0]].count())
                    return reduced2_df, restrict_reviews, limit_method, limit, sub_limit_method

                elif yn2.upper() == "N":
                    valid2 = True
                    restrict_reviews = yn2
                    print("total reviews: ", reduced_df[reduced_df.columns[0]].count())
                    return reduced_df, restrict_reviews, limit_method, limit, sub_limit_method

        elif yn.upper() == "N":
            valid = True
            limit = yn
            return df, restrict_reviews, limit_method, limit, sub_limit_method


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

    valid = False
    while not valid:
        n = input("How many users to evaluate on?: ")
        try:
            n = int(n)
            valid = True
        except:
            print("Invalid input, if you entered a number, try a smaller value")

    print("Retrieving chosen users")
    file_name = "intermediate_results/" + code.upper() + "_top_users.pkl"
    try:
        open_file = open(file_name, "rb")
        spread = pickle.load(open_file)
        open_file.close()
    except:
        users_ratings = {}
        grouped = df.groupby(['reviewerID'])
        max_count = 0
        for name, group in tqdm(grouped):
            group = group
            counts = dict(group["overall"].value_counts())
            if sum(counts.values()) > max_count:
                max_count = sum(counts.values())
            if sum(counts.values()) > 100:
                users_ratings[name] = counts

        for k, v in tqdm(users_ratings.items()):
            temp = [0, 0, 0, 0, 0]
            for k2, v2 in v.items():
                temp[k2-1] = v2/sum(v.values())
            users_ratings[k] = temp

        # calc rmse of each list
        target = [0.2, 0.2, 0.2, 0.2, 0.2]
        spread = []
        for k, v in tqdm(users_ratings.items()):
            rms = mean_squared_error(target, v, squared=False)
            spread.append([k, rms * (max_count-sum(counts.values()))])

        spread = sorted(spread, key=lambda x: x[1], reverse=True)

        open_file = open(file_name, "wb")
        pickle.dump(spread, open_file)
        open_file.close()

    reduced = spread[-n:]
    chosen = []
    for item in reduced:
        chosen.append(item[0])

    return chosen

    # if code.upper() == "K":
    #     # kindle
    #     return "A1CIS4LOWYGZGA"
    # elif code.upper() == "M":
    #     # movie
    #     return "A5TZXWU8AALIC"
    # elif code.upper() == "V":
    #     # video games
    #     return "A2582KMXLK2P06"
    # elif code.upper() == "D":
    #     # digital music
    #     return "A3W4D8XOGLWUN5"