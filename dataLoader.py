import pandas as pd 
import gzip
import json
import math
import random
from tqdm import tqdm
from os.path import exists
from personality_neighbourhood import get_neighbourhood
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

random.seed(42)


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


def reduceDF(df, df_code, chosen, restrict_reviews, limit_method, limit, sub_limit_method, first_time, g_abs_num):
    valid = False
    while not valid:
        if first_time:
            yn = "N"
        else:
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
                    yn3 = input("Personal or Non-Personal Refinement? [P/N]: ")
                if yn3.upper() == "N":
                    print("Maximum number of users: ", len(df['reviewerID'].value_counts()))
                    print(str(df['reviewerID'].nunique()))
                    valid3 = True
                    limit_method = yn3
                    # get n most common reviewers
                    valid6 = False
                    while not valid6:
                        if g_abs_num is not None:
                            valid6 = True
                            n = g_abs_num
                        else:
                            try:
                                n = int(input("Enter a number of reviewers (Recommended 5% = " + str(int(df['reviewerID'].nunique() / 100) * 5) + "): "))
                                valid6 = True
                                g_abs_num = n
                            except:
                                print("Invalid input")
                    print("Number of reviewers: ", n)

                    valid4 = False
                    while not valid4:
                        if sub_limit_method is not None:
                            yn4 = sub_limit_method
                        else:
                            yn4 = input("Stratified, Most Common, or Random? [S/M/R]: ")
                        if yn4.upper() == "S":
                            sub_limit_method = yn4
                            valid4 = True
                            reduced_df = stratified_sampling(n, df, chosen, df_code)
                        elif yn4.upper() == "M":
                            sub_limit_method = yn4
                            valid4 = True
                            reduced_df = most_common(n, df, chosen)
                        elif yn4.upper() == "R":
                            sub_limit_method = yn4
                            valid4 = True
                            reduced_df = select_random(n, df, chosen)
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
                            print()
                            print("[N] Neighbourhood")
                            print("[L] Linear Stratified")
                            print("[S] Logarithmic Stratified")
                            yn5 = input("Please select an option above: ")
                        if yn5.upper() == "S":
                            sub_limit_method = yn5
                            valid5 = True
                            stratified = True
                            neighbours_df = get_neighbourhood(chosen, df_code, stratified, 0) # 0 = log
                            neighbours = neighbours_df["reviewerID"].unique()
                            print(len(neighbours), " chosen")
                            reduced_df = df[df['reviewerID'].isin(neighbours)]
                        elif yn5.upper() == "L":
                            sub_limit_method = yn5
                            valid5 = True
                            stratified = True
                            neighbours_df = get_neighbourhood(chosen, df_code, stratified, 1) # 1 = lin
                            neighbours = neighbours_df["reviewerID"].unique()
                            print(len(neighbours), " chosen")
                            reduced_df = df[df['reviewerID'].isin(neighbours)]
                        elif yn5.upper() == "N":
                            sub_limit_method = yn5
                            valid5 = True
                            stratified = False
                            neighbours_df = get_neighbourhood(chosen, df_code, stratified, None)
                            neighbours = neighbours_df["reviewerID"].unique()
                            print(len(neighbours), " chosen")
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
                    return reduced2_df, restrict_reviews, limit_method, limit, sub_limit_method, g_abs_num

                elif yn2.upper() == "N":
                    valid2 = True
                    restrict_reviews = yn2
                    print("total reviews: ", reduced_df[reduced_df.columns[0]].count())
                    return reduced_df, restrict_reviews, limit_method, limit, sub_limit_method, g_abs_num

        elif yn.upper() == "N":
            valid = True
            limit = yn
            return df, restrict_reviews, limit_method, limit, sub_limit_method, g_abs_num


def stratified_sampling(n, df, chosen, code):
    print("Conducting Stratified Sampling...")
    k = math.floor(n/5)
    remainder = n % 5
    print("k: ", k)
    print("remainder: ", remainder)
    steps = 1/(k-1)
    new_df = pd.DataFrame()

    if code.upper() == "K":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")
    elif code.upper() == "M":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv")
    elif code.upper() == "V":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5_personality.csv")
    elif code.upper() == "D":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5_personality.csv")
    elif code.upper() == "P":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5_personality.csv")
    elif code.upper() == "G":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5_personality.csv")
    elif code.upper() == "S":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5_personality.csv")
    elif code.upper() == "C":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5_personality.csv")

    domains = ["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"]

    # for each domain, take the k/2 lowest and k/2 highest
    ids = []
    for i in range(5):
        if i == 4:
            # make sure the chosen id is counted among them
            k -= 1
            k += remainder
        large_num = math.ceil(k/2)
        small_num = math.floor(k/2)
        largest = personalities.nlargest(large_num, domains[i])["reviewerID"]
        smallest = personalities.nsmallest(small_num, domains[i])["reviewerID"]

        for item in largest:
            ids.append(item)
        for item in smallest:
            ids.append(item)
        personalities = personalities[~personalities['reviewerID'].isin(ids)]

    ids.append(chosen)
    print(len(ids))

    reduced_df = df[df['reviewerID'].isin(ids)]

    # saved_df = df.copy()
    # df = df.merge(personalities, on="reviewerID")
    #
    # new_df = df.copy()[0:0]
    #
    # domains = ["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"]
    #
    # count = 1
    # for domain in domains:
    #     print(str(count) + "/5")
    #     count += 1
    #     target = 0
    #     for i in range(int(k)):
    #         row_to_add = df.iloc[(df[domain] - target).abs().argsort()[0]]
    #         new_df = new_df.append(row_to_add)
    #         target += steps
    #
    # new_df.append(new_df)
    #
    # # get ids of chosen_users
    # ids = new_df["reviewerID"].unique()
    # ids = np.append(ids, chosen)
    # print(str(len(ids)) + " chosen users")
    #
    # print("Chosen Stratified Sample")
    # final_sample = saved_df[saved_df['reviewerID'].isin(ids)]

    return reduced_df


def most_common(n, df, chosen):
    frequents = df['reviewerID'].value_counts()[:n-1].index.tolist()

    frequents.append(chosen)
    print(str(len(frequents)) + " chosen users")

    reduced_df = df[df['reviewerID'].isin(frequents)]

    return reduced_df


def select_random(n, df, chosen):
    all_users = df['reviewerID'].unique().tolist()
    chosen_users = random.sample(all_users, n-1)
    chosen_users.append(chosen)
    print(str(len(chosen_users)) + " chosen users")

    reduced_df = df[df['reviewerID'].isin(chosen_users)]

    return reduced_df


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
            # if sum(counts.values()) >= 100:
            if sum(counts.values()) >= 1:
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
