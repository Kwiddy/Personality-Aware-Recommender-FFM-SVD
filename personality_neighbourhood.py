import pandas as pd


# function to find the users with the most similar personalities to a target user
def find_neighbours(id, threshold, df_code):

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

    sims_df = personalities_df.copy()

    # print("---------------------------------")
    # print(sims_df)

    sims_df["diff"] = abs(personalities_df["Extroversion"] - u_ext) + abs(personalities_df["Agreeableness"] - u_agr) + abs(personalities_df["conscientiousness"] - u_con) + abs(personalities_df["Neurotisicm"] - u_neu) + abs(personalities_df["Openness_to_Experience"] - u_ope)
    sims_df = sims_df.drop(columns=["Unnamed: 0", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])

    # print("---------------------------------")
    # print(sims_df)

    # the maximum difference allowed between the chosen user and each other user
    reduced_df = sims_df[sims_df['diff'] <= threshold]

    # threshold of  0.5 gives   225926 rows
    #               0.3         149293
    #               0.1         19000
    #               0.05        2169

    return reduced_df


def get_neighbourhood(user, df_code):
    happy = False
    while not happy:
        try:
            threshold = float(input("Enter a threshold value (e.g. 0.3): "))
            # print(threshold)
            # print(type(threshold))
            # print(user)
            df = find_neighbours(user, threshold, df_code)
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
