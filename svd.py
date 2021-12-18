# https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/

import pandas as pd
from numpy.linalg import svd
import numpy as np
import operator

def create_svd(original_df, ffm_df, user_ID):

    # THIS RESULTED IN AN INDEXERROR WHEN OPERATING ON THE WHOLE DATASET, THIS IS CURRENTLY AN OPEN ERROR AND REQUIRES A SMALLER DATASET
    # print(original_df)

    users_rating_df = original_df[["reviewerID", "asin", "overall"]].copy()
    # anything smaller than this results in a dataet thats too large (somewhere between 2500000 and 3000000 causes the error)
    users_rating_df = users_rating_df.iloc[:-3000000]
    # print(user_rating_df)
    print("Reduced dataframe size: ", users_rating_df.size)

    chosen_user_df = users_rating_df.loc[users_rating_df['reviewerID'] == user_ID]
    users_reviewed_items = chosen_user_df['asin'].tolist()
    print("Chosen user: ", user_ID)
    print("Reviewed items: ", users_reviewed_items)

    print("Creating SVD...")
    # create user-item review score matrix
    # review_matrix = user_rating_df.pivot(index="reviewerID", columns="asin", values="overall").fillna(0)
    review_matrix = pd.pivot_table(users_rating_df, index="reviewerID", columns="asin", values="overall").fillna(0)
    matrix = review_matrix.values

    print(review_matrix)
    # print("-----")
    # print(matrix)

    print(review_matrix.columns)
    i = 0
    map_1 = {}
    map_2 = {}
    item_scores = {}
    for id in review_matrix.columns:
        map_1[i] = id
        map_2[id] = i
        item_scores[id] = 0 

    # 1 = {i: id}
    # 1.5 = {id: i}
    # 2 = {id: score}
    # for id in user_reviewed_items, find i in 1.5, for each value j in i, find id of j using 1, and then add to score using 2

    # THIS BELOW CHUNK WAS HEAVILY TAKEN FROM THE REFERENCE AT THE TOP, AS WAS A LOT OF THIS FUNCTION
    print("Applying SVD...")
    u, s, vh = svd(matrix, full_matrices=False)
    print(vh.shape)
    print("Finding Similarities...")
    for item_id in users_reviewed_items:
        item_location = map_2[item_id]
        highest_similarity = -np.inf
        highest_sim_col = -1
        for col in range(1,vh.shape[1]):
            similarity = cosine_similarity(vh[:,item_location], vh[:,col])
            item_scores[item_id] += similarity
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_sim_col = col
    
    print("Column %d is most similar to column 0" % highest_sim_col)

    # For each item, total its cosine-similarities to each item that the user has reviewed, output the k most similar items
    sorted_similarities = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))

    top_100 = list(sorted_similarities)[:100]
    print(top_100)


def cosine_similarity(v,u):
    return (v @ u)/ (np.linalg.norm(v) * np.linalg.norm(u))
 
