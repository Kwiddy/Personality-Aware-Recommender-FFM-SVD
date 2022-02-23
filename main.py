from reviewAPR import review_APR
from dataLoader import getDF, reduceDF
from svd import create_svd
from svd2 import create_svd_2
from lgbmRegressor import create_lightgbm
from datetime import date, datetime
from evaluation import evaluate
from analysis import exploratory_analysis
from personalityapproach1 import approach1
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


def main():
    file_path, parent_path, ext, df_code = choose_data()

    retrieved_df = getDF(file_path, parent_path, ext)

    full_df, chosen_user = reduceDF(retrieved_df, df_code)

    print("Chosen user: ", chosen_user)

    # ffm_df = review_APR(full_df, parent_path, ext)

    # take a test split for the chosen_user 
    train, test = train_test_split(full_df, chosen_user)

    select_method(full_df, train, test, chosen_user, df_code)


def choose_data():
    v_choice = False
    print("[M] - Movies and TV")
    print("[D] - Digital Music")
    print("[K] - Kindle Store")
    print("[V] - Video Games")
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

    new_path = [char for char in file_path]
    i = -1
    while new_path[i] != "/":
        del new_path[i]
    parent_path = ''.join(new_path)
    new_path = parent_path + extension

    return file_path, parent_path, extension, df_code


def select_method(full_df, train, test, chosen_user, code):

    # take equal amounts of each rating
    print("Rating distribution: ", dict(full_df["overall"].value_counts()))

    # make an equal number of each case
    equal = full_df.groupby('overall').head(min(dict(full_df["overall"].value_counts()).values())).reset_index(drop=True)
    user_rows = full_df.loc[full_df['reviewerID'] == chosen_user]
    equal = pd.concat([equal, user_rows])
    print("Equal")
    print(equal)
    print("Rating distribution: ", dict(equal["overall"].value_counts()))

    valid2 = False
    while not valid2:
        choice = input("Model or Analysis? [M/A]: ")
        if choice.upper() == "M":
            valid2 = True
            valid = False
            while not valid:
                yn = input("Include personality in model? [Y/N]: ")
                if yn.upper() == "Y":
                    print("Using Personality....")
                    valid = True
                    valid_in = False
                    print("[L] - LightGBM")
                    print("[R] - Random Forest")
                    print("[S] - SVD")
                    print("[P] - SVD++")
                    print("[A] - All")
                    while not valid_in:
                        while not valid_in:
                            method = input("Please choose a method above: ")
                            if method.upper() == "L":
                                valid_in = True
                                recommendations_df = create_lightgbm(equal, train, chosen_user, "L", code, True)
                            elif method.upper() == "R":
                                valid_in = True
                                recommendations_df = create_lightgbm(equal, train, chosen_user, "R", code, True)
                            elif method.upper() == "S":
                                valid_in = True
                                recommendations_df = approach1(equal, train, chosen_user, False, code, True, None)
                            elif method.upper() == "P":
                                valid_in = True
                                recommendations_df = approach1(equal, train, chosen_user, True, code, True, None)
                            elif method.upper() == "A":
                                valid_in = True
                                # results = [LightGBM, RF, SVD, SVD++]
                                results = []
                                dp = get_dp()
                                results.append(["LightGBM", create_lightgbm(equal, train, chosen_user, "L", code, False)])
                                results.append(["RandomForest", create_lightgbm(equal, train, chosen_user, "R", code, False)])
                                results.append(["SVD", approach1(equal, train, chosen_user, False, code, False, dp)])
                                results.append(["SVD++", approach1(equal, train, chosen_user, True, code, False, dp)])
                elif yn.upper() == "N":
                    valid = True
                    # choose method
                    print("")
                    print("Methods:")
                    print("[S] - cheat SVD")
                    print("[T] - SVD")
                    print("[P] - SVD++")
                    valid_in = False
                    while not valid_in:
                        method = input("Please choose a method above: ")
                        if method.upper() == "S":
                            valid_in = True
                            # recommendations = create_svd(full_df, ffm_df, chosen_user)
                            recommendations_df = create_svd(full_df, train, chosen_user)
                        if method.upper() == "T":
                            valid_in = True
                            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
                            recommendations_df = create_svd_2(full_df, train, chosen_user, 0)
                        if method.upper() == "P":
                            valid_in = True
                            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
                            recommendations_df = create_svd_2(full_df, train, chosen_user, 1)
                else:
                    print("Invalid input, please enter a 'Y' or an 'N'")

            if yn.upper() == "Y" and method.upper() == "A":
                # results = [LightGBM, RF, SVD, SVD++]
                df_dict = defaultdict(list)
                print()
                for result in results:
                    response = evaluate(result[1], train, test, chosen_user, False)
                    df_dict["Model"].append(result[0])
                    df_dict["RMSE 1"].append(response[0])
                    df_dict["RMSE 2"].append(response[1])
                    df_dict["RMSE 3"].append(response[2])
                    df_dict["RMSE 4"].append(response[3])
                    df_dict["RMSE 5"].append(response[4])
                    df_dict["Overall RMSE"].append(response[5])
                result_df = pd.DataFrame(df_dict)
                result_df = result_df.set_index("Model")
                print(result_df)
                print()

            else:
                print("Most recommended")
                print(recommendations_df.head(10))

                response = evaluate(recommendations_df, train, test, chosen_user, True)

        elif choice.upper() == "A":
            # exploratory_analysis(full_df)
            exploratory_analysis(equal)
            valid2 = True
        else:
            print("Invalid input")

    go_again(full_df, train, test, chosen_user, code)


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


def go_again(full_df, train, test, chosen_user, code):
    valid = False
    while not valid:
        yn = input("Select a different method? [Y/N]: ")
        if yn.upper() == "Y":
            valid = True
            print()
            select_method(full_df, train, test, chosen_user, code)
        elif yn.upper() == "N":
            valid = True

    valid2 = False
    while not valid2:
        yn = input("Choose different dataset? [Y/N]: ")
        if yn.upper() == "Y":
            valid2 = True
            print()
            main()
        elif yn.upper() == "N":
            valid2 = True


def get_dp():
    try:
        return int(input("Round SVD by (Recommended: 6): "))
    except:
        get_dp()


# track runtime
start = datetime.now()
main()
print("Runtime: ", datetime.now()-start)