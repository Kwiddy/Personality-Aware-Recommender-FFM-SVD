import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
tqdm.pandas()


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

    personalities_df = pd.read_csv(path)

    user_row = personalities_df.loc[personalities_df['reviewerID'] == id]
    u_ext = user_row["Extroversion"].tolist()[0]
    u_agr = user_row["Agreeableness"].tolist()[0]
    u_con = user_row["conscientiousness"].tolist()[0]
    u_neu = user_row["Neurotisicm"].tolist()[0]
    u_ope = user_row["Openness_to_Experience"].tolist()[0]
    target_user = [u_ext, u_agr, u_con, u_neu, u_ope]

    sims_df = personalities_df.copy()

    # print("---------------------------------")
    # print(sims_df)

    # change old measure to rmse
    sims_df["diff"] = abs(personalities_df["Extroversion"] - u_ext) + abs(personalities_df["Agreeableness"] - u_agr) + abs(personalities_df["conscientiousness"] - u_con) + abs(personalities_df["Neurotisicm"] - u_neu) + abs(personalities_df["Openness_to_Experience"] - u_ope)

    # sims_df["diff"] = sims_df.progress_apply(lambda x: get_rmse(x.Extroversion, x.Agreeableness, x.conscientiousness, x.Neurotisicm, x.Openness_to_Experience, target_user), axis=1)
    # sims_df["diff"] = math.sqrt(((personalities_df["Extroversion"] - u_ext)**2 + (personalities_df["Agreeableness"] - u_agr)**2 + (personalities_df["conscientiousness"] - u_con)**2 + (personalities_df["Neurotisicm"] - u_neu)**2 + (personalities_df["Openness_to_Experience"] - u_ope)**2)/5)

    # print("Calculating individual errors...")
    # sims_df["ExtErr2"] = (personalities_df["Extroversion"] - u_ext)**2
    # sims_df["AgrErr2"] = (personalities_df["Agreeableness"] - u_ext) ** 2
    # sims_df["conErr2"] = (personalities_df["conscientiousness"] - u_ext) ** 2
    # sims_df["NeuErr2"] = (personalities_df["Neurotisicm"] - u_ext) ** 2
    # sims_df["OteErr2"] = (personalities_df["Openness_to_Experience"] - u_ext) ** 2
    # print("Calculating mean...")
    # sims_df["Mean"] = (sims_df["ExtErr2"] + sims_df["AgrErr2"] + sims_df["conErr2"] + sims_df["NeuErr2"] + sims_df["OteErr2"]) / 5
    # print("Calculating root...")
    # sims_df["diff"] = np.sqrt(sims_df["Mean"])

    sims_df = sims_df.drop(columns=["Unnamed: 0", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])

    return sims_df


def get_neighbourhood(user, df_code):
    happy = False
    sims_df = find_neighbours(user, df_code)
    while not happy:
        try:
            # threshold = float(input("Enter a threshold value (e.g. 0.3): "))
            # df = sims_df[sims_df['diff'] <= threshold]

            # get top 5% of most similar users
            df = sims_df.nsmallest((int(sims_df[sims_df.columns[0]].count() / 100)*5), 'diff')

            print("Number of rows: ", df[df.columns[0]].count())
            # print(df)
            # exit()
            valid = False
            while not valid:
                inp = input("Is this a valid number of rows? [Y/N]: ")
                if inp.upper() == "Y":
                    valid = True
                    happy = True
                elif inp.upper() == "N":
                    valid = True
                else:
                    print("Please enter a 'Y' or 'N'")
        except:
            print("Invalid input, must be a float")
    return df


# user_id = "A2M1CU2IRZG0K9"
# # user_id = "A1JLU5H1CCENWX"
# # A1JLU5H1CCENWX
# find_neighbours(user_id, 0.05, "M")
