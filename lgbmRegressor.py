import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
import sklearn


def translate(val, dic):
    return dic[val]


def pre_process(df, asin_convert):
    df = df.drop(
        columns=["Unnamed: 0_x", "Unnamed: 0_y", "verified", "reviewTime", "style", "image", "reviewerName",
                 "reviewText", "summary", "vote"])
    df = df.drop(columns=["reviewerID"])

    df["asin"] = df.progress_apply(lambda x: translate(x.asin, asin_convert), axis=1)

    return df


def create_lightgbm(full_df, train, chosen_user, model_choice, code):
    print("full_df")
    print(full_df)
    print()
    print("train")
    print(train)
    print()
    print("Chosen user")
    print(chosen_user)

    if code.upper() == "K":
        personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")
    elif code.upper() == "M":
        personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv")
    elif code.upper() == "V":
        personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5_personality.csv")
    elif code.upper() == "D":
        personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5_personality.csv")

    R = 42
    target_col = "overall"

    print(full_df)
    full_df = full_df.merge(personalities, on="reviewerID")
    print(full_df)

    ######################################################
    # # originally: p0.11 = 1.578, p0.11 = 1.530
    # # MIGHT BE WRONG
    # print(full_df["Extroversion"].min())
    # print(full_df["Extroversion"].max())
    # print(full_df)
    # # Normalize each of the personality columns -- THIS MIGHT BE WRONG
    # minval = full_df["Extroversion"].min()
    # maxval = full_df["Extroversion"].max()
    # full_df["Extroversion"] = full_df.progress_apply(lambda x: normalize(x.Extroversion, minval, maxval), axis=1)
    # minval = full_df["Agreeableness"].min()
    # maxval = full_df["Agreeableness"].max()
    # full_df["Agreeableness"] = full_df.progress_apply(lambda x: normalize(x.Agreeableness, minval, maxval), axis=1)
    # minval = full_df["conscientiousness"].min()
    # maxval = full_df["conscientiousness"].max()
    # full_df["conscientiousness"] = full_df.progress_apply(lambda x: normalize(x.conscientiousness, minval, maxval), axis=1)
    # minval = full_df["Neurotisicm"].min()
    # maxval = full_df["Neurotisicm"].max()
    # full_df["Neurotisicm"] = full_df.progress_apply(lambda x: normalize(x.Neurotisicm, minval, maxval), axis=1)
    # minval = full_df["Openness_to_Experience"].min()
    # maxval = full_df["Openness_to_Experience"].max()
    # full_df["Openness_to_Experience"] = full_df.progress_apply(lambda x: normalize(x.Openness_to_Experience, minval, maxval),
    #                                                  axis=1)
    # print(full_df["Extroversion"].min())
    # print(full_df["Extroversion"].max())
    #######################################################


    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]
    print("---------------------")
    print(neighbourhood)
    print(target)


    print(neighbourhood.columns)

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

    neighbourhood = pre_process(neighbourhood, asin_convert)
    target = pre_process(target, asin_convert)

    x_target = target.drop(columns=target_col)
    y_target = target[target_col]

    print(neighbourhood.columns)
    print(target.columns)

    x = neighbourhood.drop(columns=target_col)
    y = neighbourhood[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          test_size=0.4,
                                                          random_state=R)

    if model_choice == "L":
        model = lgb.LGBMRegressor(n_estimators=100, class_weight="balanced")
    elif model_choice == "R":
        model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # predictions = model.predict(x_test)
    #
    # print(predictions)
    #
    # # Generate a classification report and other metrics to determin performance
    # print("MSE: %.3f" % sklearn.metrics.mean_squared_error(y_test, predictions))
    # print("RMSE: %.3f" % sklearn.metrics.mean_squared_error(y_test, predictions, squared=False))

    predictions = model.predict(x_target)

    print(predictions)

    print()
    print("LightGBM Feature importances: ")
    hd = list(x_train.columns)
    for i, f in zip(hd, model.feature_importances_):
        print(i, round(f * 100, 2))
    print()

    result = x_target.copy()
    print(y_target)
    print(x_target)
    print(predictions)
    result["predictions"] = predictions
    result = result.drop(columns=["unixReviewTime", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])
    result = result.sort_values(by=['predictions'], ascending=False)

    result["asin"] = result.progress_apply(lambda j: translate(j.asin, reverse_asin_convert), axis=1)

    return result


def normalize(value, min, max):
    return (value-min) / (max-min)