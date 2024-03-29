# This file includes the neural network regression model for recommendation generation

# imports
from pandas import read_csv
from keras.models import Sequential
import scikeras
from keras.layers import Dense, Dropout
from keras.callbacks import History
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from lgbmRegressor import translate, pre_process
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# some parts of this function were adapted from tutorial:
#   https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
def baseline_nn(full_df, train, test, chosen_user, code, disp, split, n_epochs, use_personality):

    # load personality data
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

    # separate target user from other users
    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]

    # create train and test splits from the user's data
    train_target = target.loc[target["asin"].isin(train)]
    test_target = target.loc[target["asin"].isin(test)]

    # add user training data to the neighbourhood
    neighbourhood = pd.concat([neighbourhood, train_target])
    target = test_target.copy()

    # convert asins to numerical values
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

    # pre-process dataframes
    neighbourhood = pre_process(neighbourhood, asin_convert, use_personality)
    target = pre_process(target, asin_convert, use_personality)

    # define df for training features and target features
    x_target = target.drop(columns=target_col)
    y_target = target[target_col]

    x = neighbourhood.drop(columns=target_col)
    y = neighbourhood[target_col]

    # determine outputting of model
    if disp:
        vb = 1
    else:
        vb = 0

    # define model
    estimator = KerasRegressor(build_fn=baseline_model(use_personality), epochs=n_epochs, batch_size=1, verbose=vb, random_state=R)

    # prepare data
    sc = StandardScaler()
    x = sc.fit_transform(x)
    x_target_save = x_target.copy()
    x_target = sc.transform(x_target)

    # introduce history for graphing
    history = estimator.fit(x, y)

    # generate predictions
    preds = estimator.predict(x_target)
    predictions = []
    for item in preds:
        predictions.append(float(item))
    result = x_target_save.copy()
    result["predictions"] = predictions

    # sort personality columns if relevant
    if use_personality:
        result = result.drop(columns=["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm",
                                      "Openness_to_Experience"])

    # sort by predictions
    result = result.sort_values(by=['predictions'], ascending=False)

    # translate asins back to original values
    result["asin"] = result.progress_apply(lambda j: translate(j.asin, reverse_asin_convert), axis=1)

    # graph epochs-loss
    losses = history.history_['loss']
    epoch_num = [x for x in range(len(losses))]
    plt.plot(epoch_num, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("saved_results/baseline_nn_learn.png")
    plt.close()

    return result


def baseline_model(use_personality):
    if use_personality:
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=6, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    else:
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=1, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model