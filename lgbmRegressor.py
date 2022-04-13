# This file contains the LightGBM and random forest predictors

# imports
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
import sklearn
import random
import math

# set random seet
random.seed(42)


# translate from a given dictionary
def translate(val, dic):
    return dic[val]


# normalize value
def normalize(value, minv, maxv):
    return (value-minv) / (maxv-minv)


# pre-process data by cleaning a given dataframe
def pre_process(df, asin_convert, use_personality):
    if use_personality:
        df = df.drop(
            columns=["Unnamed: 0_x", "Unnamed: 0_y", "verified", "reviewTime", "style", "image", "reviewerName",
                     "reviewText", "summary", "vote", "unixReviewTime"])
    else:
        df = df.drop(
            columns=["Unnamed: 0", "verified", "reviewTime", "style", "image", "reviewerName", "reviewText", "summary",
                     "vote", "unixReviewTime"])

    df = df.drop(columns=["reviewerID"])

    df["asin"] = df.progress_apply(lambda x: translate(x.asin, asin_convert), axis=1)

    return df


# create lightgbm and random forest model
def create_lightgbm(full_df, train, test, chosen_user, model_choice, code, disp, split, use_personality):

    # load personality data if relevant
    if use_personality:
        if code.upper() == "K":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")
        elif code.upper() == "M":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv")
        elif code.upper() == "V":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5_personality.csv")
        elif code.upper() == "D":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5_personality.csv")
        elif code.upper() == "P":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5_personality.csv")
        elif code.upper() == "G":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5_personality.csv")
        elif code.upper() == "S":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5_personality.csv")
        elif code.upper() == "C":
            personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5_personality.csv")
        full_df = full_df.merge(personalities, on="reviewerID")

    R = 42
    target_col = "overall"

    # define user neighbourhood and target user
    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]

    # create train and target data
    train_target = target.loc[target["asin"].isin(train)]
    test_target = target.loc[target["asin"].isin(test)]
    neighbourhood = pd.concat([neighbourhood, train_target])
    target = test_target.copy()

    # convert asins to numerical ids
    asins = full_df.asin.unique()
    reverse_asin_convert = {}
    id = 0
    for item in asins:
        reverse_asin_convert[id] = item
        id += 1
    asin_convert = {}
    id = 0
    for item in asins:
        asin_convert[item] = id
        id += 1

    # pre-process data to reduce unnecessary features
    neighbourhood = pre_process(neighbourhood, asin_convert, use_personality)
    target = pre_process(target, asin_convert, use_personality)

    # create train-test split
    x_target = target.drop(columns=target_col)
    y_target = target[target_col]
    x = neighbourhood.drop(columns=target_col)
    y = neighbourhood[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          test_size=0.4,
                                                          random_state=R)

    # define model to be used
    if model_choice == "L":
        model = lgb.LGBMRegressor(random_state=R)
    elif model_choice == "R":
        model = RandomForestRegressor(random_state=R)

    # construct grid search and fit
    param_grid = create_grid(model_choice)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(x_train, y_train)

    grid_used = True

    # output best parameters
    print()
    print("Best parameters: ")
    print(grid_search.best_params_)

    to_predict = x_target.set_index("asin").copy()
    to_predict = to_predict[~to_predict.index.duplicated(keep='first')].reset_index()

    predictions = grid_search.predict(x_target)

    # output feature importance if relevant
    if disp and not grid_used:
        print()
        print("LightGBM Feature importances: ")
        hd = list(x_train.columns)
        for i, f in zip(hd, model.feature_importances_):
            print(i, round(f * 100, 2))
        print()

    # clean and sort results
    result = x_target.copy()
    result["predictions"] = predictions
    if use_personality:
        result = result.drop(columns=["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm",
                                      "Openness_to_Experience"])
    result = result.sort_values(by=['predictions'], ascending=False)

    result["asin"] = result.progress_apply(lambda j: translate(j.asin, reverse_asin_convert), axis=1)

    filtered_result = result.set_index("asin").copy()
    filtered_result = filtered_result[~filtered_result.index.duplicated(keep='first')].reset_index()

    return filtered_result

# grid search, comments show values tested, these have been commented out to save computation time
def create_grid(approach):
    if approach == "L":
        # num_leaves = [31, 91, 121, 151, 251, 191, 211]
        num_leaves = [201]
        # n_estimators = [100, 200, 300, 400, 500, 700, 600, 650]
        n_estimators = [550]
        # class_weight = ["balanced", None]
        class_weight = [None]
        # learning_rate = [0.1, 0.2, 0.3]
        learning_rate = [0.3]
        # min_child_weight = [0.0001, 0.001, 0.01]
        min_child_weight = [0]
        # min_child_samples = [5, 10,20,30]
        min_child_samples = [0]
        # subsample_for_bin = [100000, 200000, 300000]
        subsample_for_bin = [100000]
        param_grid = {
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "class_weight": class_weight,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "min_child_samples": min_child_samples,
            "subsample_for_bin": subsample_for_bin
        }
    else:
        param_grid = {}

    return param_grid