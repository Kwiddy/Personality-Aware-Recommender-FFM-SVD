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
        i += 1

    # 1 = {i: id}
    # 1.5 = {id: i}
    # 2 = {id: score}
    # for id in user_reviewed_items, find i in 1.5, for each value j in i, find id of j using 1, and then add to score using 2

    # THIS BELOW CHUNK WAS HEAVILY TAKEN FROM THE REFERENCE AT THE TOP, AS WAS A LOT OF THIS FUNCTION
    print("Applying SVD...")
    u, s, vh = svd(matrix, full_matrices=False)
    print(vh.shape)
    sim_to_items = []
    print("Finding Similarities...")
    for item_id in users_reviewed_items:
        item_location = map_2[item_id]
        highest_similarity = -np.inf
        highest_sim_col = -1
        for col in range(1,vh.shape[1]):
            similarity = cosine_similarity(vh[:,item_location], vh[:,col])
            # print(similarity)
            # find id of item being compared 
            compared_id = map_1[col]
            item_scores[compared_id] += similarity
            # print(item_scores[item_id])
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_sim_col = col
    
        # print("Column %d is most similar to column 0" % highest_sim_col)

        # print(item_scores)
        # For each item, total its cosine-similarities to each item that the user has reviewed, output the k most similar items
        # sorted_similarities = dict( sorted(item_scores.items(), key=lambda item: item[1], reverse=True))

        to_return = []
        for item in list(item_scores):
            to_return.append([item, item_scores[item]])

        # # print("")
        # # print(sorted_similarities)
        # top_100 = list(sorted_similarities)[:101]

        # to_return = []
        # for item in top_100:
        #     to_return.append([item, sorted_similarities[item]])

        # discard the most similar item - this will be the item itself
        del to_return[0]

        specific_entry = chosen_user_df.loc[chosen_user_df['asin'] == item_id]
        sim_to_items.append([item_id, specific_entry["overall"], to_return])

    results = svd_predictions(sim_to_items)    

    return to_return


def svd_predictions(inp):
    # for each item in inp, 
    #   item[0] is the reviewed item that everything else is being compared to
    #   item[1] is the list of all other items and their similarity to that item

    # totals_n = {}
    # for each item in inp
    #    score = rating given to item[0]
    #    for each sim in item[1]:
    #       if sim[0] not in totals_n then totals_n[sim[0]] == 0
    #       totals_n[sim[0]] += score * sim[1]
    #
    # totals_d = {}
    # for each item in inp:
    #   for each sim in item[1]
    #       if sim[0] not in totals_d then totals_d[sim[0]] == 0
    #       totals_d[sim[0]] += abs(sim[1])

    # for each item in totals_n, divide by corresponding value in totals_d, output score

    # try and guess the target score for each item
    print(inp)
    print(inp[0][0])
    print(inp[0][1])

    totals_n = {}
    for item in inp:
        id = item[0]
        score = item[1]
        for sim in item[2]:
            if sim[0] not in totals_n:
                totals_n[sim[0]] = 0 
            totals_n[sim[0]] += score * sim[1]
        
    totals_d = {}
    for item in inp:
        for sim in item[2]:
            if sim[0] not in totals_d:
                totals_d[sim[0]] = 0 
            totals_d[sim[0]] += abs(sim[1])

    predictions = []
    for key, value in totals_n.items():
        n = value.item()
        d = totals_d[key].item()
        result = n / d
        print(n)
        print(d)
        print(result)
        exit()
        # making the result item[0] so I can sort them easily
        predictions.append([result, key])
        # print(n)
        # print(d)
        # print(result)
        # print(predictions)
        # exit()

    # # sort the predictions
    sorted_predictions = sorted(predictions, reverse=True)
    print(sorted_predictions)
    # print(predictions)
    # print(type(predictions))



    exit()


def cosine_similarity(v,u):
    return (v @ u)/ (np.linalg.norm(v) * np.linalg.norm(u))
 
