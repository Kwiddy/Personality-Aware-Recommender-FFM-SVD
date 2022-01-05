from reviewAPR import review_APR
from dataLoader import getDF, reduceDF
from svd import create_svd
from svd2 import create_svd_2
from datetime import date, datetime
from evaluation import evaluate
import pandas as pd


def main():
    # track runtime 
    start = datetime.now()

    file_path, parent_path = choose_data()

    retrieved_df = getDF(file_path, parent_path)

    full_df, chosen_user = reduceDF(retrieved_df)

    # FOR EACH ROW IN DATA, INPUT ONLY THE USER_ID AND THE REVIEW
    # ffm_df = review_APR(full_df, parent_path)

    # take a test split for the chosen_user 
    train, test = train_test_split(full_df, chosen_user)

    select_method(full_df, train, test, chosen_user)

    print("Runtime: ", datetime.now()-start)


def choose_data():

    v_choice = False
    print("[M] - Movies and TV")
    while not v_choice:
        choice = input("Please enter one of the datasets above: ")
        if choice.upper() == "M":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movies_and_TV_5.json.gz'
            extension = "Movie_and_TV_5.csv"
            v_choice = True

    new_path = [char for char in file_path]
    i = -1
    while new_path[i] != "/":
        del new_path[i]
    parent_path = ''.join(new_path)
    new_path = parent_path + extension

    return file_path, parent_path


def select_method(full_df, train, test, chosen_user):
    yn = input("Include personality? [Y/N]: ")
    if yn.upper() == "Y":
        print("Using Personality....")
        ########
        
    # choose method
    print("")
    print("Methods:")
    print("[S] - cheat SVD")
    print("[T] - SVD")
    print("[P] - SVD++")
    valid_in = False
    while not valid_in:
        method = input("Please choose a method above: ")
        if method.upper() == "S":
            valid_in = True
            # recommendations = create_svd(full_df, ffm_df, chosen_user)
            recommendations_df = create_svd(full_df, train, chosen_user)
        if method.upper() == "T":
            valid_in = True
            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
            recommendations_df = create_svd_2(full_df, train, chosen_user, 0)
        if method.upper() == "P":
            valid_in = True
            # recommendations = create_svd_2(full_df, ffm_df, chosen_user)
            recommendations_df = create_svd_2(full_df, train, chosen_user, 1)

    print(recommendations_df.head(10))
    
    evaluate(recommendations_df, train, test, chosen_user)

    go_again(full_df, train, test, chosen_user)


def train_test_split(df, user):

    users = df["reviewerID"].unique()

    parts_train = []
    parts_test = []

    for i in range(len(users)):
        user_df = df.loc[df['reviewerID'] == users[i]]
        rows, columns = user_df.shape

        split = 0.2
        splitter = int(rows * split)

        test = user_df.iloc[splitter:]
        parts_test.append(test)

        train = user_df.iloc[:splitter]
        parts_train.append(train)


    res_train = pd.concat(parts_train)
    res_test = pd.concat(parts_test)

    return res_train, res_test


def go_again(full_df, train, test, chosen_user):
    valid = False
    while not valid:
        yn = input("Select a different method? [Y/N]: ")
        if yn.upper() == "Y":
            valid = True
            print()
            select_method(full_df, train, test, chosen_user)
        elif yn.upper() == "N":
            valid = True

main()