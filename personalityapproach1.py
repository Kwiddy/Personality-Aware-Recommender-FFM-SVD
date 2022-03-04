
from surprise import Reader, Dataset, SVD, SVDpp
import numpy as np
import pandas as pd
from tqdm import tqdm


def approach1(full_df, train, chosen_user, plus_bool, code, disp, dp):
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

    full_df = full_df.merge(personalities, on="reviewerID")

    R = 42

    valid_dp = False
    if dp is None:
        while not valid_dp:
            try:
                dp = int(input("SVD - Round by (recommended: 5): "))
                valid_dp = True
            except:
                print("Invalid - Enter an integer")

    full_df["Extroversion"] = full_df["Extroversion"].round(dp)
    full_df["Openness_to_Experience"]= full_df["Openness_to_Experience"].round(dp)
    full_df["Agreeableness"] = full_df["Agreeableness"].round(dp)
    full_df["conscientiousness"] = full_df["conscientiousness"].round(dp)
    full_df["Neurotisicm"] = full_df["Neurotisicm"].round(dp)

    # reduce
    small_df = full_df[["reviewerID", "asin", "overall"]].copy()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    # get the list of the movie ids
    unique_ids = small_df['asin'].unique()
    # get the list of the ids that the user has rated
    seen_ids = train.loc[train['reviewerID'] == chosen_user, 'asin']
    # remove the rated movies for the recommendations
    items_to_predict = np.setdiff1d(unique_ids, seen_ids)

    if plus_bool:
        algo = SVDpp(random_state=R, verbose=True)
    else:
        algo = SVD(random_state=R, verbose=True)

    algo.fit(data.build_full_trainset())
    my_recs1 = []
    for iid in items_to_predict:
        my_recs1.append((iid, algo.predict(uid=chosen_user, iid=iid).est))

    chosen_peronsality_row = personalities.loc[personalities['reviewerID'] == chosen_user]
    ext_score = round(float(chosen_peronsality_row["Extroversion"]), dp)
    ote_score = round(float(chosen_peronsality_row["Openness_to_Experience"]), dp)
    agr_score = round(float(chosen_peronsality_row["Agreeableness"]), dp)
    con_Score = round(float(chosen_peronsality_row["conscientiousness"]), dp)
    neu_score = round(float(chosen_peronsality_row["Neurotisicm"]), dp)

    corr_ext, my_recs2 = personality_svd("Extroversion", full_df, items_to_predict, ext_score, plus_bool, R)
    corr_ote, my_recs3 = personality_svd("Openness_to_Experience", full_df, items_to_predict, ote_score, plus_bool, R)
    corr_agr, my_recs4 = personality_svd("Agreeableness", full_df, items_to_predict, agr_score, plus_bool, R)
    corr_con, my_recs5 = personality_svd("conscientiousness", full_df, items_to_predict, con_Score, plus_bool, R)
    corr_neu, my_recs6 = personality_svd("Neurotisicm", full_df, items_to_predict, neu_score, plus_bool, R)

    map_corr = (corr_con + corr_agr + corr_ote + corr_neu + corr_ext) / 5
    print("Mean Absolute Personality (MAP) Correlation: ", map_corr)

    # ####################################

    my_recs = []

    # give each personality SVD an importance equal to its correlation to the overall score to 2dp
    for i in range(len(my_recs1)):
        results = []
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
        for j in range(100):
            results.append(my_recs1[i][1])
        mean = sum(results) / len(results)
        my_recs.append((my_recs1[i][0], mean))

    result = pd.DataFrame(my_recs, columns=['asin', 'predictions']).sort_values('predictions', ascending=False)

    return result, dp


def personality_svd(personality, full_df, items_to_predict, user_score, plus_bool, R):
    # reduce
    small_df = full_df[[personality, "asin", "overall"]].copy()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    if plus_bool:
        algo = SVDpp(random_state=R, verbose=True)
    else:
        algo = SVD(random_state=R, verbose=True)

    algo.fit(data.build_full_trainset())
    recs = []
    for iid in items_to_predict:
        recs.append((iid, algo.predict(uid=user_score, iid=iid).est))

    correlation = abs(full_df[personality].corr(full_df["overall"]))

    return correlation, recs
