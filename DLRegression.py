
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


# adapted from tutorial https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
def baseline_nn(full_df, train, chosen_user, code, disp):
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

    R = 42
    target_col = "overall"

    full_df = full_df.merge(personalities, on="reviewerID")
    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]
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

    x = neighbourhood.drop(columns=target_col)
    y = neighbourhood[target_col]

    # estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
    # kfold = KFold(n_splits=10)
    if disp:
        vb = 1
    else:
        vb = 0
    # estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1, verbose=vb, random_state=42)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=25, batch_size=1, verbose=vb, random_state=R)

    # kfold = KFold(n_splits=2)
    # results = cross_val_score(estimator, x, y, cv=kfold)
    # print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)
    x_target_save = x_target.copy()
    x_target = sc.transform(x_target)

    history = estimator.fit(x, y)
    preds = estimator.predict(x_target)

    predictions = []
    for item in preds:
        predictions.append(float(item))

    result = x_target_save.copy()
    result["predictions"] = predictions
    result = result.drop(columns=["unixReviewTime", "Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm",
                                  "Openness_to_Experience"])
    result = result.sort_values(by=['predictions'], ascending=False)

    result["asin"] = result.progress_apply(lambda j: translate(j.asin, reverse_asin_convert), axis=1)

    losses = history.history_['loss']
    epoch_num = [x for x in range(len(losses))]

    plt.plot(epoch_num, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("saved_results/baseline_nn_learn.png")
    plt.close()

    return result


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model