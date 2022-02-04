import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn


def translate(val, dic):
    return dic[val]

def pre_process(df):
    df = df.drop(
        columns=["Unnamed: 0_x", "Unnamed: 0_y", "verified", "reviewTime", "style", "image", "reviewerName",
                 "reviewText", "summary", "vote"])
    df = df.drop(columns=["reviewerID"])

    asins = df.asin.unique()

    asin_convert = {}
    id = 0
    for item in asins:
        asin_convert[item] = id
        id += 1

    df["asin"] = df.progress_apply(lambda x: translate(x.asin, asin_convert), axis=1)

    return df


def create_lightgbm(full_df, train, chosen_user):
    print("full_df")
    print(full_df)
    print()
    print("train")
    print(train)
    print()
    print("Chosen user")
    print(chosen_user)

    personalities = pd.read_csv("Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")


    R = 42
    target_col = "overall"

    print(full_df)
    full_df = full_df.merge(personalities, on="reviewerID")
    print(full_df)

    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]
    print("---------------------")
    print(neighbourhood)
    print(target)


    print(neighbourhood.columns)

    neighbourhood = pre_process(neighbourhood)
    target = pre_process(target)

    x_target = target.drop(columns=target_col)
    y_target = target[target_col]

    print(neighbourhood.columns)
    print(target.columns)

    x = neighbourhood.drop(columns=target_col)
    y = neighbourhood[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          test_size=0.4,
                                                          random_state=R)

    model = lgb.LGBMRegressor(n_estimators=100, class_weight="balanced")
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

    # Generate a classification report and other metrics to determin performance
    print("MSE: %.3f" % sklearn.metrics.mean_squared_error(y_target, predictions))
    print("RMSE: %.3f" % sklearn.metrics.mean_squared_error(y_target, predictions, squared=False))



    print()
    print("LightGBM Feature importances: ")
    hd = list(x_train.columns)
    for i, f in zip(hd, model.feature_importances_):
        print(i, round(f * 100, 2))
    print()
