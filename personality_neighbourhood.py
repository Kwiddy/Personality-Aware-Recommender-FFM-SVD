from dataLoader import getDF, reduceDF
import pandas as pd


# function to find the users with the most similar personalities to a target user
def find_neighbours(id):
    personalities_df = pd.read_csv("./Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv")

    user_row = personalities_df.loc[personalities_df['reviewerID'] == id]
    u_ext = user_row["Extroversion"][0]
    u_agr = user_row["Agreeableness"][0]
    u_con = user_row["conscientiousness"][0]
    u_neu = user_row["Neurotisicm"][0]
    u_ope = user_row["Openness_to_Experience"][0]

    sims_df = personalities_df.copy()

    sims_df["diff"] = abs(personalities_df["Extroversion"] - u_ext) + abs(personalities_df["Agreeableness"] - u_agr) + abs(personalities_df["conscientiousness"] - u_con) + abs(personalities_df["Neurotisicm"] - u_neu) + abs(personalities_df["Openness_to_Experience"] - u_ope)
    sims_df = sims_df.drop(columns=["Unnamed: 0", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])

    # the maximum difference allowed between the chosen user and each other user
    threshold = 0.05

    reduced_df = sims_df[sims_df['diff'] <= threshold]

    # threshold of  0.5 gives   225926 rows
    #               0.3         149293
    #               0.1         19000
    #               0.05        2169

    print(reduced_df)


user_id = "A2M1CU2IRZG0K9"
find_neighbours(user_id)
