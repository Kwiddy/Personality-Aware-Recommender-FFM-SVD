# Taken and adjusted from
# https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65

# and 
# https://predictivehacks.com/how-to-run-recommender-systems-in-python/ 

import pandas as pd
import numpy as np
import scipy
from scipy.linalg import sqrtm

from surprise import NMF, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, CoClustering
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset

def simple_svd(full_df, train, chosen_user, items_to_predict, data):
    print()    
    algo = SVD()
    algo.fit(data.build_full_trainset())
    my_recs = []
    for iid in items_to_predict:
        my_recs.append((iid, algo.predict(uid=chosen_user,iid=iid).est))
        
    res = pd.DataFrame(my_recs, columns=['asin', 'predictions']).sort_values('predictions', ascending=False)

    return res

def create_svd_2(full_df, train, chosen_user, svd_bit):

    # reduce
    small_df = full_df[["reviewerID", "asin", "overall"]].copy()
    small_train = train[["reviewerID", "asin", "overall"]].copy()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(small_df, reader)

    # get the list of the movie ids
    unique_ids = small_df['asin'].unique()
    # get the list of the ids that the user has rated
    seen_ids = train.loc[train['reviewerID']==chosen_user, 'asin']
    # remove the rated movies for the recommendations
    items_to_predict = np.setdiff1d(unique_ids, seen_ids)

    # print(res.head(10))
    if svd_bit == 0:
        result = simple_svd(full_df, train, chosen_user, items_to_predict, data)

    return result

# # def create_svd_2(full_df, ffm_df, chosen_user):
# def create_svd_2(full_df, chosen_user):

#     data= full_df

#     # data = full_df.iloc[:-3000000]
#     # print(user_rating_df)
#     print("Reduced dataframe size: ", data.size)

#     users = data['reviewerID'].unique()
#     items = data['asin'].unique() 

#     print("Making splits...")
#     test = pd.DataFrame(columns=data.columns)
#     train = pd.DataFrame(columns=data.columns)
#     test_ratio = 0.2 
#     for u in users:
#         temp = data[data['reviewerID'] == u]
#         n = len(temp)
#         test_size = int(test_ratio*n)
#     temp = temp.sort_values('unixReviewTime').reset_index()
#     temp.drop('index', axis=1, inplace=True)
        
#     dummy_test = temp.loc[n-1-test_size :]
#     dummy_train = temp.loc[: n-2-test_size]
        
#     test = pd.concat([test, dummy_test])
#     train = pd.concat([train, dummy_train])

#     # to test the performance over a different number of features
#     no_of_features = [8,10,12,14,17]
#     print("Creating Utility Matrix")
#     utilMat, users_index, items_index = create_utility_matrix(train)
#     print("Starting SVD for feature nums...")
#     for f in no_of_features: 
#         svdout = svd(utilMat, k=f)
#         pred = [] #to store the predicted ratings
#         for _,row in test.iterrows():
#             user = row['reviewerID']
#             item = row['asin']
#             u_index = users_index[user]
#             if item in items_index:
#                 i_index = items_index[item]
#                 pred_rating = svdout[u_index, i_index]
#             else:
#                 pred_rating = np.mean(svdout[u_index, :])
#             pred.append(pred_rating)

#     print("Calculating RMSE: ")
#     print(rmse(test['overall'], pred))


#     return "hi"

# def svd(train, k):
#     utilMat = np.array(train)
#     # the nan or unavailable entries are masked
#     mask = np.isnan(utilMat)
#     masked_arr = np.ma.masked_array(utilMat, mask)
#     item_means = np.mean(masked_arr, axis=0)
#     # nan entries will replaced by the average rating for each item
#     utilMat = masked_arr.filled(item_means)
#     x = np.tile(item_means, (utilMat.shape[0],1))
#     # we remove the per item average from all entries.
#     # the above mentioned nan entries will be essentially zero now
#     utilMat = utilMat - x
#     # The magic happens here. U and V are user and item features
#     U, s, V=np.linalg.svd(utilMat, full_matrices=False)
#     s=np.diag(s)
#     # we take only the k most significant features
#     s=s[0:k,0:k]
#     U=U[:,0:k]
#     V=V[0:k,:]
#     s_root=sqrtm(s)
#     Usk=np.dot(U,s_root)
#     skV=np.dot(s_root,V)
#     UsV = np.dot(Usk, skV)
#     UsV = UsV + x
#     print("svd done")
#     return UsV

# def rmse(true, pred):
#     # this will be used towards the end
#     x = true - pred
#     return sum([xi*xi for xi in x])/len(x)

# def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):
#     """
#         :param data:      Array-like, 2D, nx3
#         :param formatizer:pass the formatizer
#         :return:          utility matrix (n x m), n=users, m=items
#     """
        
#     itemField = formatizer['item']
#     userField = formatizer['user']
#     valueField = formatizer['value']
#     userList = data.loc[:,userField].tolist()
#     itemList = data.loc[:,itemField].tolist()
#     valueList = data.loc[:,valueField].tolist()
#     users = list(set(data.loc[:,userField]))
#     items = list(set(data.loc[:,itemField]))
#     users_index = {users[i]: i for i in range(len(users))}
#     pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
#     for i in range(0,len(data)):
#         item = itemList[i]
#         user = userList[i]
#         value = valueList[i]
#     pd_dict[item][users_index[user]] = value
#     X = pd.DataFrame(pd_dict)
#     X.index = users
        
#     itemcols = list(X.columns)
#     items_index = {itemcols[i]: i for i in range(len(itemcols))}
#     # users_index gives us a mapping of user_id to index of user
#     # items_index provides the same for items
#     return X, users_index, items_index