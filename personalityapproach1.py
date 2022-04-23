# the svd usage in this approach has been adapted from the example found in
# https://predictivehacks.com/how-to-run-recommender-systems-in-python/

# imports
from surprise import Reader, Dataset, SVD, SVDpp
import pandas as pd
import random

# set random seed
random.seed(42)


# FFM-SVD and SVD approach
def approach1(full_df, train, test, chosen_user, plus_bool, code, disp, dp, use_personality, split):
    # load relevant dataset
    if code.upper() == "K":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")
    elif code.upper() == "M":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movie_and_TV_5_personality.csv")
    elif code.upper() == "V":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5_personality.csv")
    elif code.upper() == "D":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5_personality.csv")
    elif code.upper() == "P":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5_personality.csv")
    elif code.upper() == "G":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5_personality.csv")
    elif code.upper() == "S":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5_personality.csv")
    elif code.upper() == "C":
        personalities = pd.read_csv(
            "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5_personality.csv")
    full_df = full_df.merge(personalities, on="reviewerID")

    # calculate MAP correlations
    pers_domains = ["Extroversion", "Openness_to_Experience", "Agreeableness", "conscientiousness", "Neurotisicm"]
    abs_corrs = []
    for personality in pers_domains:
        abs_corrs.append(abs(full_df[personality].corr(full_df["overall"])))
    map_corr = sum(abs_corrs) / len(abs_corrs)
    print("MAP Correlation: ", map_corr)
    print("MAP Sum: ", sum(abs_corrs))

    R = 42

    # choose bucketing precision
    valid_dp = False
    if dp is None:
        while not valid_dp:
            try:
                dp = int(input("SVD - Round by (recommended: 5): "))
                valid_dp = True
            except:
                print("Invalid - Enter an integer")

    full_df["Extroversion"] = full_df["Extroversion"].round(dp)
    full_df["Openness_to_Experience"] = full_df["Openness_to_Experience"].round(dp)
    full_df["Agreeableness"] = full_df["Agreeableness"].round(dp)
    full_df["conscientiousness"] = full_df["conscientiousness"].round(dp)
    full_df["Neurotisicm"] = full_df["Neurotisicm"].round(dp)

    # reduce
    small_df = full_df[["reviewerID", "asin", "overall"]].copy()

    seen_items = small_df.loc[small_df['reviewerID'] == chosen_user, 'asin'].tolist()
    test_items = test
    train_items = train

    # remove test items
    small_df = small_df.drop(small_df[(small_df['asin'].isin(test_items)) & (small_df['reviewerID'] == chosen_user)].index)

    # load data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    # choose verbosity
    vb = True

    ## original parameters
    # id_importance = 1
    # n_factors = 100
    # n_epochs = 20
    # init_mean = 0
    # init_std_dev = 0.1
    # lr_all = 0.005
    # reg_all = 0.02

    id_importance = 1
    n_factors = 100
    n_epochs = 20
    init_mean = 0
    init_std_dev = 0.2
    lr_all = 0.0062
    reg_all = 0.01

    # create model
    if plus_bool:
        algo = SVDpp(random_state=R, verbose=vb, n_factors=n_factors, n_epochs=n_epochs, init_mean=init_mean, init_std_dev=init_std_dev, lr_all=lr_all, reg_all=reg_all)
    else:
        algo = SVD(random_state=R, verbose=vb, n_factors=n_factors, n_epochs=n_epochs, init_mean=init_mean, init_std_dev=init_std_dev, lr_all=lr_all, reg_all=reg_all)

    # fit model
    algo.fit(data.build_full_trainset())

    # predict test items
    my_recs1 = []
    for iid in test_items:
        my_recs1.append((iid, algo.predict(uid=chosen_user, iid=iid).est))

    # create other 5 SVDs if FFM-SVD
    if use_personality:
        chosen_peronsality_row = personalities.loc[personalities['reviewerID'] == chosen_user]
        ext_score = round(float(chosen_peronsality_row["Extroversion"]), dp)
        ote_score = round(float(chosen_peronsality_row["Openness_to_Experience"]), dp)
        agr_score = round(float(chosen_peronsality_row["Agreeableness"]), dp)
        con_Score = round(float(chosen_peronsality_row["conscientiousness"]), dp)
        neu_score = round(float(chosen_peronsality_row["Neurotisicm"]), dp)

        corr_ext, my_recs2 = personality_svd("Extroversion", full_df, test_items, ext_score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all)
        corr_ote, my_recs3 = personality_svd("Openness_to_Experience", full_df, test_items, ote_score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all)
        corr_agr, my_recs4 = personality_svd("Agreeableness", full_df, test_items, agr_score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all)
        corr_con, my_recs5 = personality_svd("conscientiousness", full_df, test_items, con_Score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all)
        corr_neu, my_recs6 = personality_svd("Neurotisicm", full_df, test_items, neu_score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all)

        # determine the ID importance
        id_importance = 2.2 * sum([corr_ote, corr_ext, corr_con, corr_neu, corr_agr])

        ## this is now ouputted elsewhere
        # map_corr = (corr_con + corr_agr + corr_ote + corr_neu + corr_ext) / 5
        # print("Mean Absolute Personality (MAP) Correlation: ", map_corr)


    my_recs = []

    # give each personality SVD an importance equal to its correlation to the overall score to 2dp
    for i in range(len(my_recs1)):
        results = []
        if use_personality:
            for j in range(int(100*round(float(corr_ext), 2))):
                results.append(my_recs2[i][1])
            for j in range(int(100*round(float(corr_ote), 2))):
                results.append(my_recs3[i][1])
            for j in range(int(100*round(float(corr_agr), 2))):
                results.append(my_recs4[i][1])
            for j in range(int(100*round(float(corr_con), 2))):
                results.append(my_recs5[i][1])
            for j in range(int(100*round(float(corr_neu), 2))):
                results.append(my_recs6[i][1])
        for j in range(int(100*round(float(id_importance), 2))):
            results.append(my_recs1[i][1])
        mean = sum(results) / len(results)
        my_recs.append((my_recs1[i][0], mean))

    # save results
    result = pd.DataFrame(my_recs, columns=['asin', 'predictions']).sort_values('predictions', ascending=False)

    return result, dp


# function to create a personality-trait SVD
def personality_svd(personality, full_df, items_to_predict, user_score, plus_bool, R, chosen_user, vb, n_factors, n_epochs, init_mean, init_std_dev, lr_all, reg_all):
    small_df = full_df[[personality, "reviewerID", "asin", "overall"]].copy()
    small_df = small_df.drop(small_df[(small_df['asin'].isin(items_to_predict)) & (small_df['reviewerID'] == chosen_user)].index)
    small_df = small_df[[personality, "asin", "overall"]]

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    if plus_bool:
        algo = SVDpp(random_state=R, verbose=vb, n_factors=n_factors, n_epochs=n_epochs, init_mean=init_mean, init_std_dev=init_std_dev, lr_all=lr_all, reg_all=reg_all)
    else:
        algo = SVD(random_state=R, verbose=vb, n_factors=n_factors, n_epochs=n_epochs, init_mean=init_mean, init_std_dev=init_std_dev, lr_all=lr_all, reg_all=reg_all)

    algo.fit(data.build_full_trainset())
    recs = []
    for iid in items_to_predict:
        recs.append((iid, algo.predict(uid=user_score, iid=iid).est))

    correlation = abs(full_df[personality].corr(full_df["overall"]))

    return correlation, recs
