import math
import pandas as pd

def evaluate(results, train, test, user, display):
    # get the test split for the chosen user
    user_test = test.loc[test['reviewerID'] == user]
    user_test_relev = user_test[["asin", "overall"]].copy()

    # rename overall to actual
    user_test_relev = user_test_relev.rename(columns={"overall": "actual"})

    # set asin as index for both
    user_test_relev = user_test_relev.set_index('asin')
    results = results.set_index('asin')

    comparison = user_test_relev.join(results)

    result = calc_rmse(comparison, display)

    return result


def calc_rmse(df, disp):
    rmse_df = df.copy()
    rmse_df["RMSE"] = (df["actual"]-df["predictions"])**2
    
    rmse = math.sqrt(rmse_df["RMSE"].mean())

    if disp:
        print(rmse_df)
        print()
        print("RMSE: ", rmse)

    # RMSE for each different scoring
    rmse_list = [[], [], [], [], []]
    for index, row in rmse_df.iterrows():
        rmse_list[int(row["actual"])-1].append(float(row["RMSE"]))

    if disp:
        print("1 Rating RMSE: ", math.sqrt(sum(rmse_list[0]) / len(rmse_list[0])))
        print("2 Rating RMSE: ", math.sqrt(sum(rmse_list[1]) / len(rmse_list[1])))
        print("3 Rating RMSE: ", math.sqrt(sum(rmse_list[2]) / len(rmse_list[2])))
        print("4 Rating RMSE: ", math.sqrt(sum(rmse_list[3]) / len(rmse_list[3])))
        print("5 Rating RMSE: ", math.sqrt(sum(rmse_list[4]) / len(rmse_list[4])))
        print()

    # [1 RMSE, 2 RMSE, 3 RMSE, 4 RMSE, 5 RMSE, rmse]
    return [math.sqrt(sum(rmse_list[0]) / len(rmse_list[0])), math.sqrt(sum(rmse_list[1]) / len(rmse_list[1])),
            math.sqrt(sum(rmse_list[2]) / len(rmse_list[2])), math.sqrt(sum(rmse_list[3]) / len(rmse_list[3])),
            math.sqrt(sum(rmse_list[4]) / len(rmse_list[4])), rmse]