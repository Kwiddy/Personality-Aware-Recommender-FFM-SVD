from reviewAPR import review_APR
from dataLoader import getDF, reduceDF, find_chosen
from svd import create_svd
from svd2 import create_svd_2
from lgbmRegressor import create_lightgbm
from datetime import date, datetime
from evaluation import evaluate, global_eval
from analysis import exploratory_analysis
from personalityapproach1 import approach1
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from DLRegression import baseline_nn
from os.path import exists

# Initialise globals
pers_yes = None
model_analysis = None
method_choice = None
dp_round = None
restrict_reviews = None
limit = None
limit_method = None
sub_limit_method = None
g_results = []
g_all = False
g_test_split = 0.3
g_absolute_num = None


def main():
    global restrict_reviews
    global limit_method
    global sub_limit_method
    global limit
    global g_absolute_num

    file_path, parent_path, ext, df_code = choose_data()

    retrieved_df = getDF(file_path, parent_path, ext)

    chosen_user = find_chosen(retrieved_df, df_code)
    print("Chosen users: ", chosen_user)

    for chosen in chosen_user:
        personality_path = parent_path + ext[:-4] + "_personality.csv"

        if exists(personality_path):
            first_time = False
            full_df, rr, lm, lim, slm, abs_num = reduceDF(retrieved_df, df_code, chosen, restrict_reviews, limit_method, limit,
                                                 sub_limit_method, first_time, g_absolute_num)
            restrict_reviews = rr
            limit_method = lm
            sub_limit_method = slm
            limit = lim
            g_absolute_num = abs_num

        else:
            first_time = True
            full_df, rr, lm, lim, slm, abs_num = reduceDF(retrieved_df, df_code, chosen, restrict_reviews, limit_method, limit,
                                                 sub_limit_method, first_time, g_absolute_num)
            ffm_df = review_APR(full_df, parent_path, ext)
            print("New dataset personalities computed - Please rerun the program")
            exit()

        # take a test split for the chosen_user
        train, test = train_test_split(full_df, chosen_user)

        select_method(full_df, train, test, chosen, df_code)

    output_results()


def choose_data():
    v_choice = False
    print("[M] - Movies and TV")
    print("[D] - Digital Music")
    print("[K] - Kindle Store")
    print("[V] - Video Games")
    print("[P] - Pet Supplies")
    print("[G] - Patio, Lawn, & Garden")
    print("[S] - Sports & Outdoors")
    print("[C] - Music (CDs & Vinyl)")
    while not v_choice:
        choice = input("Please enter one of the datasets above: ")
        if choice.upper() == "M":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movies_and_TV_5.json.gz'
            extension = "Movie_and_TV_5.csv"
            v_choice = True
            df_code = "M"
        elif choice.upper() == "D":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5.json.gz'
            extension = "Digital_Music_5.csv"
            v_choice = True
            df_code = "D"
        elif choice.upper() == "K":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5.json.gz'
            extension = "Kindle_Store_5.csv"
            v_choice = True
            df_code = "K"
        elif choice.upper() == "V":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5.json.gz'
            extension = "Video_Games_5.csv"
            v_choice = True
            df_code = "V"
        elif choice.upper() == "P":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5.json.gz'
            extension = "Pet_Supplies_5.csv"
            v_choice = True
            df_code = "P"
        elif choice.upper() == "G":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5.json.gz'
            extension = "Patio_Lawn_and_Garden_5.csv"
            v_choice = True
            df_code = "G"
        elif choice.upper() == "S":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5.json.gz'
            extension = "Sports_and_Outdoors_5.csv"
            v_choice = True
            df_code = "S"
        elif choice.upper() == "C":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5.json.gz'
            extension = "CDs_and_Vinyl_5.csv"
            v_choice = True
            df_code = "C"

    new_path = [char for char in file_path]
    i = -1
    while new_path[i] != "/":
        del new_path[i]
    parent_path = ''.join(new_path)
    new_path = parent_path + extension

    return file_path, parent_path, extension, df_code


def select_method(full_df, train, test, chosen_user, code):
    global pers_yes
    global model_analysis
    global method_choice
    global dp_round
    global g_results
    global g_all
    global g_test_split
    global g_absolute_num

    # make an equal number of each case
    equal = full_df.groupby('overall').head(min(dict(full_df["overall"].value_counts()).values())).reset_index(drop=True)
    user_rows = full_df.loc[full_df['reviewerID'] == chosen_user]
    equal = pd.concat([equal, user_rows])
    print("Rating distribution: ", dict(equal["overall"].value_counts()))

    # the number of features in each method for adjusted r**2 metric
    feature_nums = {1: 6, 2: 6, 3: 6, 4: 6, 5: 1, 6: 6}

    valid2 = False
    while not valid2:
        if model_analysis is not None:
            choice = model_analysis
        else:
            choice = input("Model or Analysis? [M/A]: ")
        if choice.upper() == "M":
            model_analysis = choice
            valid2 = True
            valid = False
            while not valid:
                if pers_yes is not None:
                    yn = pers_yes
                else:
                    yn = input("Include personality in model / Do All? [Y/N/A]: ")
                if yn.upper() == "Y":
                    pers_yes = yn
                    print("Using Personality....")
                    valid = True
                    valid_in = False
                    if method_choice is None:
                        print("[L] - LightGBM")
                        print("[R] - Random Forest")
                        print("[S] - SVD")
                        print("[P] - SVD++")
                        print("[N] - Neural Network")
                    while not valid_in:
                        while not valid_in:
                            if method_choice is not None:
                                method = method_choice
                            else:
                                method = input("Please choose a method above: ")
                            if method.upper() == "L":
                                method_choice = method
                                valid_in = True
                                recommendations_df = create_lightgbm(equal, train, chosen_user, "L", code, True, g_test_split)
                                m_name = "LightGBM"
                                p_type = True
                                b_type = True
                                m_choice = 1
                            elif method.upper() == "R":
                                method_choice = method
                                valid_in = True
                                recommendations_df = create_lightgbm(equal, train, chosen_user, "R", code, True, g_test_split)
                                m_name = "RandomForest"
                                m_choice = 2
                                p_type = True
                                b_type = True
                            elif method.upper() == "S":
                                method_choice = method
                                valid_in = True
                                recommendations_df, dp_result = approach1(equal, train, chosen_user, False, code, True, dp_round, True, g_test_split)
                                print(recommendations_df)
                                dp_round = dp_result
                                m_name = "6-SVD"
                                m_choice = 3
                                p_type = True
                                b_type = True
                            elif method.upper() == "P":
                                method_choice = method
                                valid_in = True
                                recommendations_df, dp_result = approach1(equal, train, chosen_user, True, code, True, dp_round, True, g_test_split)
                                dp_round = dp_result
                                m_name = "6-SVD++"
                                p_type = True
                                b_type = True
                                m_choice = 4
                            elif method.upper() == "N":
                                method_choice = method
                                valid_in = True
                                m_choice = 6
                                recommendations_df = baseline_nn(equal, train, chosen_user, code, True, g_test_split)
                                m_name = "NeuralNet"
                                p_type = True
                                b_type = True
                elif yn.upper() == "N":
                    valid = True
                    pers_yes = yn
                    # choose method
                    print("")
                    if method_choice is None:
                        print("Methods:")
                        # print("[S] - cheat SVD")
                        print("[T] - SVD")
                        print("[P] - SVD++")
                    valid_in = False
                    while not valid_in:
                        if method_choice is not None:
                            method = method_choice
                        else:
                            method = input("Please choose a method above: ")
                        if method.upper() == "S":
                            method_choice = method
                            valid_in = True
                            # recommendations = create_svd(full_df, ffm_df, chosen_user)
                            recommendations_df, dp_result = approach1(full_df, train, chosen_user, False, code, True, dp_round, False, g_test_split)
                            m_name = "SVD"
                            p_type = False
                            b_type = False
                        if method.upper() == "T":
                            method_choice = method
                            valid_in = True
                            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
                            recommendations_df, dp_result = approach1(full_df, train, chosen_user, True, code, True, dp_round, False, g_test_split)
                            m_name = "SVD"
                            p_type = False
                            b_type = False
                            m_choice = 5
                        if method.upper() == "P":
                            method_choice = method
                            valid_in = True
                            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
                            recommendations_df = create_svd_2(full_df, train, chosen_user, 1)
                            m_name = "SVD++"
                            m_choice = 5
                            p_type = False
                            b_type = False

                elif yn.upper() == "A":
                    valid = True
                    pers_yes = yn
                    g_all = True
                    # results = [LightGBM, RF, SVD, SVD++]
                    results = []

                    if dp_round is not None:
                        dp = dp_round
                    else:
                        valid_dp = False
                        while not valid_dp:
                            try:
                                dp = int(input("Round SVD by (Recommended: 5): "))
                                dp_round = dp
                                valid_dp = True
                            except:
                                print("Invalid - Please enter an integer")

                    print("Personality LightGBM...")
                    results.append(
                        ["LightGBM", True, True, create_lightgbm(equal, train, chosen_user, "L", code, False, g_test_split), 1])
                    # print(results)
                    # results.append(
                    #     ["LightGBM", True, False, create_lightgbm(full_df, train, chosen_user, "L", code, False, g_test_split), 1])
                    # print("Personality Random Forest...")
                    # results.append(
                    #     ["RandomForest", True, True, create_lightgbm(equal, train, chosen_user, "R", code, False, g_test_split), 2])
                    # results.append(
                    #     ["RandomForest", True, False, create_lightgbm(full_df, train, chosen_user, "R", code, False, g_test_split),
                    #      2])
                    print("Personality 6-SVD...")
                    results.append(
                        ["6-SVD", True, True, approach1(equal, train, chosen_user, False, code, False, dp, True, g_test_split)[0], 3])
                    # results.append(
                        # ["6-SVD", True, False, approach1(full_df, train, chosen_user, False, code, False, dp, True, g_test_split)[0], 3])
                    # print("Non-Personality SVD...")
                    # results.append(["SVD", False, False, approach1(full_df, train, chosen_user, False, code, False, dp, False, g_test_split)[0], 5])
                    # results.append(["SVD", False, True, approach1(equal, train, chosen_user, False, code, False, dp, False, g_test_split)[0], 5])
                    # print("Personality 6-SVD++...")
                    # results.append(
                    #     ["6-SVD++", True, True, approach1(equal, train, chosen_user, True, code, False, dp, True, g_test_split)[0], 4])
                    # results.append(
                        # ["6-SVD++", True, False, approach1(full_df, train, chosen_user, True, code, False, dp, True, g_test_split)[0], 4])
                    # print("Non-Personality SVD++...")
                    # results.append(["SVD++", False, False, approach1(full_df, train, chosen_user, True, code, False, dp, False, g_test_split)[0], 5])
                    # results.append(["SVD++", False, True, approach1(equal, train, chosen_user, True, code, False, dp, False, g_test_split)[0], 5])
                    # print("Baseline NeuralNet...")
                    # results.append(
                    #     ["Baseline NeuralNet", True, True, baseline_nn(equal, train, chosen_user, code, False, g_test_split), 6])
                else:
                    print("Invalid input, please enter a 'Y' or an 'N'")

            if yn.upper() == "A":
                # results = [LightGBM, RF, SVD, SVD++]
                df_dict = defaultdict(list)

                print()
                for result in results:
                    # print(result)
                    response = evaluate(code, result[0], result[1], result[2], result[3], train, test, chosen_user, False, feature_nums[result[4]])
                    df_dict["Model"].append(result[0])
                    df_dict["Personality"].append(result[1])
                    df_dict["Ratings-balanced"].append(result[2])
                    df_dict["RMSE 1"].append(response[0])
                    df_dict["RMSE 2"].append(response[1])
                    df_dict["RMSE 3"].append(response[2])
                    df_dict["RMSE 4"].append(response[3])
                    df_dict["RMSE 5"].append(response[4])
                    df_dict["Overall RMSE"].append(response[5])
                    df_dict["MAE"].append(response[7])
                    df_dict["Adjusted R2"].append(response[6])
                    df_dict["Prediction StD"].append(response[8])
                result_df = pd.DataFrame(df_dict)
                formatted_df = result_df.copy()
                formatted_df = formatted_df.set_index("Model")

                g_results.append(result_df)

                if code == "M":
                    prefix = "Movies"
                elif code == "D":
                    prefix = "Music"
                elif code == "K":
                    prefix = "Kindle"
                elif code == "V":
                    prefix = "Video_Games"
                elif code == "P":
                    prefix = "Pet_Supplies"
                elif code == "G":
                    prefix = "Patio_Lawn_Garden"
                elif code == "S":
                    prefix = "Sports_and_Outdoors"
                elif code == "C":
                    prefix = "CDs_and_Vinyl"
                formatted_df.to_csv("saved_results/" + prefix + "_results.csv")
                print(formatted_df)

                resultant_dfs = []
                for result in results:
                    resultant_dfs.append([result[0], result[3], result[1], result[2]])

                global_eval(result_df, resultant_dfs, test, chosen_user)

                print()

            else:
                print("Most recommended")
                print(recommendations_df.head(10))
                response = evaluate(code, m_name, p_type, b_type, recommendations_df, train, test, chosen_user, True, feature_nums[m_choice])
                g_results.append(response)

        elif choice.upper() == "A":
            # exploratory_analysis(full_df)
            model_analysis = choice
            exploratory_analysis(equal)
            valid2 = True
        else:
            print("Invalid input")


def train_test_split(df, user):

    users = df["reviewerID"].unique()

    parts_train = []
    parts_test = []

    for i in tqdm(range(len(users))):
        user_df = df.loc[df['reviewerID'] == users[i]]
        rows, columns = user_df.shape

        split = 0.2
        splitter = int(rows * split)

        test = user_df.iloc[splitter:]
        parts_test.append(test)

        train = user_df.iloc[:splitter]
        parts_train.append(train)

    res_train = pd.concat(parts_train)
    res_test = pd.concat(parts_test)

    return res_train, res_test


def output_results():
    global g_results
    global g_all

    if not g_all:
        df_dict = defaultdict(list)

        for result in g_results:
            df_dict["RMSE 1"].append(result[0])
            df_dict["RMSE 2"].append(result[1])
            df_dict["RMSE 3"].append(result[2])
            df_dict["RMSE 4"].append(result[3])
            df_dict["RMSE 5"].append(result[4])
            df_dict["Overall RMSE"].append(result[5])
            df_dict["MAE"].append(result[7])
            df_dict["Adjusted R2"].append(result[6])
            df_dict["Prediction StD"].append(result[8])
        result_df = pd.DataFrame(df_dict)
        formatted_df = result_df.copy()
        print(formatted_df)

    else:
        combined_results = pd.concat(g_results)
        combined_results = combined_results.set_index(["Model", "Personality", "Ratings-balanced"])
        print(combined_results)
        print()
        grouped_df = combined_results.groupby(combined_results.index)

        # average_df = pd.DataFrame(columns=list(g_results[0]))
        # print(average_df)
        average_df = grouped_df.mean()
        average_df = average_df.reset_index()
        # average_df.loc[average_df['Personality'] > 0, 'Personality'] = True
        # average_df.loc[average_df['Personality'] == 0, 'Personality'] = False
        # average_df.loc[average_df['Ratings-balanced'] > 0, 'Ratings-balanced'] = True
        # average_df.loc[average_df['Ratings-balanced'] == 0, 'Ratings-balanced'] = False
        average_df[['Model', 'Personality', 'Ratings-balanced']] = pd.DataFrame(average_df['index'].tolist(), index=average_df.index)
        average_df = average_df.drop('index', axis=1)
        cols = average_df.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        average_df = average_df[cols]
        print(average_df)



main()