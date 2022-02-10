
from surprise import Reader, Dataset, SVD, SVDpp
import numpy as np
import pandas as pd

def approach1(full_df, train, chosen_user):
    personalities = pd.read_csv(
        "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")

    R = 42

    full_df = full_df.merge(personalities, on="reviewerID")

    # # round columns to 2 dp
    # full_df["Extroversion"] = full_df["Extroversion"].round(2)
    # full_df["Openness_to_Experience"]= full_df["Openness_to_Experience"].round(2)

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

    algo = SVD(random_state=R)
    # algo = SVDpp()

    print()
    algo.fit(data.build_full_trainset())
    my_recs1 = []
    for iid in items_to_predict:
        my_recs1.append((iid, algo.predict(uid=chosen_user, iid=iid).est))

    print(my_recs1[0])

    chosen_peronsality_row = personalities.loc[personalities['reviewerID'] == chosen_user]
    ext_score = float(chosen_peronsality_row["Extroversion"])
    ote_score = float(chosen_peronsality_row["Openness_to_Experience"])
    # ####################################
    # print(full_df.columns)

    # reduce
    small_df = full_df[["Extroversion", "asin", "overall"]].copy()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    algo = SVD(random_state=R)

    print()
    algo.fit(data.build_full_trainset())
    my_recs2 = []
    for iid in items_to_predict:
        my_recs2.append((iid, algo.predict(uid=ext_score, iid=iid).est))
    print(my_recs2[0])
    # ####################################
    # reduce
    small_df = full_df[["Openness_to_Experience", "asin", "overall"]].copy()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    algo = SVD(random_state=R)

    print()
    algo.fit(data.build_full_trainset())
    my_recs3 = []
    for iid in items_to_predict:
        my_recs3.append((iid, algo.predict(uid=ote_score, iid=iid).est))
    print(my_recs3[0])
    # ####################################
    my_recs = []
    print(len(my_recs1))
    print(len(my_recs2))
    for i in range(len(my_recs1)):
        results = [my_recs1[i][1], my_recs2[i][1], my_recs3[i][1]]
        mean = sum(results) / len(results)
        my_recs.append((my_recs1[i][0], mean))

    result = pd.DataFrame(my_recs, columns=['asin', 'predictions']).sort_values('predictions', ascending=False)

    return result