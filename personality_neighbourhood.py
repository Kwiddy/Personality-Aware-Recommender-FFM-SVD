# imports
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import random

# initialise tqdm for progress tracking and seed for reproducibility
tqdm.pandas()
random.seed(42)


# calculate rmse
def get_rmse(ext, agr, con, neu, ope, target):
    profile = [ext, agr, con, neu, ope]
    return math.sqrt(mean_squared_error(target, profile))


# function to find the users with the most similar personalities to a target user
def find_neighbours(id, df_code):
    if df_code.upper() == "K":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv"
    elif df_code.upper() == "M":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv"
    elif df_code.upper() == "V":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5_personality.csv"
    elif df_code.upper() == "D":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5_personality.csv"
    elif df_code.upper() == "P":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5_personality.csv"
    elif df_code.upper() == "G":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5_personality.csv"
    elif df_code.upper() == "S":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5_personality.csv"
    elif df_code.upper() == "C":
        path = "./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5_personality.csv"

    personalities_df = pd.read_csv(path)

    user_row = personalities_df.loc[personalities_df['reviewerID'] == id]
    u_ext = user_row["Extroversion"].tolist()[0]
    u_agr = user_row["Agreeableness"].tolist()[0]
    u_con = user_row["conscientiousness"].tolist()[0]
    u_neu = user_row["Neurotisicm"].tolist()[0]
    u_ope = user_row["Openness_to_Experience"].tolist()[0]
    target_user = [u_ext, u_agr, u_con, u_neu, u_ope]

    sims_df = personalities_df.copy()

    # find absolute different to target user
    sims_df["diff"] = abs(personalities_df["Extroversion"] - u_ext) + abs(personalities_df["Agreeableness"] - u_agr) + abs(personalities_df["conscientiousness"] - u_con) + abs(personalities_df["Neurotisicm"] - u_neu) + abs(personalities_df["Openness_to_Experience"] - u_ope)

    sims_df = sims_df.drop(columns=["Unnamed: 0", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])

    return sims_df


# function to perform targeted pre-filtering techniqued
# log_lin 0 = log, log_lin 1 = lin
def get_neighbourhood(user, df_code, stratified, log_lin):

    sims_df = find_neighbours(user, df_code)

    # personality neighbourhood approach
    if not stratified:
        # get top 5% of most similar users
        df = sims_df.nsmallest((int(sims_df[sims_df.columns[0]].count() / 100) * 5), 'diff')
        print(str(df[df.columns[0]].count()) + " users chosen")

    # LinSP and LogSP approach
    else:
        # get percentage size of the dataframe
        target_size = int(sims_df[sims_df.columns[0]].count() / 100) * 5

        # 1/2 total within nearest bracket, 1/4 total in both sides of next bracket, 1/8 total in both sides of next...
        divisions = math.floor(math.log(target_size, 2))
        chosen = [user]
        added = 0
        max_diff = sims_df["diff"].max()
        min_diff = sims_df["diff"].min()
        for i in range(1, divisions):
            bracket_max = (i + 1) * ((max_diff - min_diff) / divisions)
            bracket_min = i * ((max_diff - min_diff) / divisions)
            bracket_df = sims_df.loc[((sims_df["diff"] >= bracket_min) & (sims_df["diff"] <= bracket_max))].copy()

            if log_lin == 0:
                # log bracket
                required_size = math.floor(target_size / 2**(i+1))
            else:
                # lin bracket
                required_size = math.floor(target_size / divisions)

            ids = bracket_df["reviewerID"].tolist()
            if required_size >= len(ids):
                for item in ids:
                    chosen.append(item)
                    added += 1
            else:
                selected = random.sample(ids, required_size)
                for item in selected:
                    chosen.append(item)
                    added += 1

        i = 0
        bracket_max = (i + 1) * ((max_diff - min_diff) / divisions)
        bracket_min = i * ((max_diff - min_diff) / divisions)

        bracket_df = sims_df.loc[((sims_df["diff"] >= bracket_min) & (sims_df["diff"] <= bracket_max))].copy()

        required_size = target_size - added - 1 # to make sure the correct number of users are added
        ids = bracket_df["reviewerID"].tolist()
        if required_size >= len(ids):
            for item in ids:
                chosen.append(item)
        else:
            selected = random.sample(ids, required_size)
            for item in selected:
                chosen.append(item)

        df = sims_df.loc[sims_df["reviewerID"].isin(chosen)]
    return df
