# imports
from reviewAPR import review_APR
from dataLoader import getDF, reduceDF, find_chosen
from lgbmRegressor import create_lightgbm
from evaluation import evaluate, global_eval
from analysis import exploratory_analysis
from personalityapproach1 import approach1
import pandas as pd
import random
from collections import defaultdict
from DLRegression import baseline_nn
from os.path import exists
from sklearn.model_selection import train_test_split

# Initialise globals
pers_yes = None
model_analysis = None
method_choice = None
dp_round = None
restrict_reviews = None
limit = None
limit_method = None
sub_limit_method = None
g_results = []
g_all = False
g_test_split = 0.3
g_absolute_num = None
g_test_bucket = None

# define random seed
random.seed(42)

# main function orchestrate pipeline of the system
def main():
    # use select global variables
    global restrict_reviews
    global limit_method
    global sub_limit_method
    global limit
    global g_absolute_num
    global g_test_split

    # select and retrieve relevant data
    file_path, parent_path, ext, df_code = choose_data()
    retrieved_df, cleaned_df = getDF(file_path, parent_path, ext) # retrieved is full_df, cleaned is pre-processed df

    # calculate skewness of data
    r_dist = dict(retrieved_df["overall"].value_counts())
    sum_vals = sum([v for v in r_dist.values()])
    for k, v in r_dist.items():
        r_dist[k] = (v/sum_vals) * 100
    skewness = 0
    a = 0
    b = 0
    for k, v in r_dist.items():
        a += (v-(sum(r_dist.values())/5))**3
        b += (v-(sum(r_dist.values())/5))**2
    skewness = round((0.2 * a) / ((0.2*b)**(3/2)), 3)
    print("Skewness: ", skewness)

    # retrieve the chosen user(s)
    chosen_user = find_chosen(cleaned_df, df_code)
    print("Chosen users: ", chosen_user)

    # iterate through users if relevant for evaluation
    i = 0
    for chosen in chosen_user:
        i += 1
        print("-----")
        print("User ", i)
        print("-----")
        # retrieve user personality
        personality_path = parent_path + ext[:-4] + "_personality.csv"

        # load pre-saved results
        if exists(personality_path):
            first_time = False
            # reduce the data using pre-filtering
            full_df, rr, lm, lim, slm, abs_num = reduceDF(cleaned_df, df_code, chosen, restrict_reviews, limit_method, limit,
                                                 sub_limit_method, first_time, g_absolute_num)

            # update global variables so that the same inputs are not required for subsequent users
            restrict_reviews = rr
            limit_method = lm
            sub_limit_method = slm
            limit = lim
            g_absolute_num = abs_num

        else:
            first_time = True
            # operate on full_df at this stage so that all reviews, even identical user-item pairs, are taken into
            #       account in review APR
            full_df, rr, lm, lim, slm, abs_num = reduceDF(retrieved_df, df_code, chosen, restrict_reviews, limit_method, limit,
                                                 sub_limit_method, first_time, g_absolute_num)
            ffm_df = review_APR(full_df, parent_path, ext)
            print("New dataset personalities computed - Please rerun the program")
            exit()

        # take a test split for the chosen_user
        user_items = full_df.loc[full_df['reviewerID'] == chosen]
        user_items = user_items["asin"].unique()
        train, test = train_test_split(user_items, test_size=g_test_split, random_state=42)

        # Proceed to select recommedner technique
        select_method(full_df, train, test, chosen, df_code, i)

    output_results()


# select domain and load
def choose_data():
    v_choice = False
    print("[M] - Movies and TV")
    print("[K] - Kindle Store")
    print("[V] - Video Games")
    print("[P] - Pet Supplies")
    print("[G] - Patio, Lawn, & Garden")
    print("[S] - Sports & Outdoors")
    print("[C] - Music (CDs & Vinyl)")
    print("[D] - Digital Music (DISCOUNTED)")
    while not v_choice:
        choice = input("Please enter one of the datasets above: ")
        if choice.upper() == "M":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movies_and_TV_5.json.gz'
            extension = "Movie_and_TV_5.csv"
            v_choice = True
            df_code = "M"
        elif choice.upper() == "D":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Digital_Music_5.json.gz'
            extension = "Digital_Music_5.csv"
            v_choice = True
            df_code = "D"
        elif choice.upper() == "K":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5.json.gz'
            extension = "Kindle_Store_5.csv"
            v_choice = True
            df_code = "K"
        elif choice.upper() == "V":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Video_Games_5.json.gz'
            extension = "Video_Games_5.csv"
            v_choice = True
            df_code = "V"
        elif choice.upper() == "P":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Pet_Supplies_5.json.gz'
            extension = "Pet_Supplies_5.csv"
            v_choice = True
            df_code = "P"
        elif choice.upper() == "G":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Patio_Lawn_and_Garden_5.json.gz'
            extension = "Patio_Lawn_and_Garden_5.csv"
            v_choice = True
            df_code = "G"
        elif choice.upper() == "S":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Sports_and_Outdoors_5.json.gz'
            extension = "Sports_and_Outdoors_5.csv"
            v_choice = True
            df_code = "S"
        elif choice.upper() == "C":
            file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/CDs_and_Vinyl_5.json.gz'
            extension = "CDs_and_Vinyl_5.csv"
            v_choice = True
            df_code = "C"

    new_path = [char for char in file_path]
    i = -1
    while new_path[i] != "/":
        del new_path[i]
    parent_path = ''.join(new_path)
    new_path = parent_path + extension

    return file_path, parent_path, extension, df_code


# select recommender technique
def select_method(full_df, train, test, chosen_user, code, user_num):
    # global variables
    global pers_yes
    global model_analysis
    global method_choice
    global dp_round
    global g_results
    global g_all
    global g_test_split
    global g_absolute_num
    global g_test_bucket

    # make an equal number of each case
    counts = dict(full_df["overall"].value_counts())
    to_del = []
    for k, v in counts.items():
        if k % 1 == 0:
            pass
        else:
            to_del.append(k)
    for key in to_del:
        del counts[key]

    # create testing (test_target) and training (neighbourhood), by splitting the target user's items
    neighbourhood = full_df[full_df.reviewerID != chosen_user]
    target = full_df[full_df.reviewerID == chosen_user]
    train_target = target.loc[target["asin"].isin(train)]
    test_target = target.loc[target["asin"].isin(test)]
    print("Len train target: ", len(train_target))
    print("Len test target: ", len(test_target))
    neighbourhood = pd.concat([neighbourhood, train_target])

    # data balancing - Equal oversampling
    original_rating = [1.0, 2.0, 3.0, 4.0, 5.0]
    rd = dict(neighbourhood["overall"].value_counts())
    o_rd = dict(full_df["overall"].value_counts())
    # print("Neighbourhood Rating distribution: ", rd)
    # print("Overall Rating distribution: ", o_rd)
    most_common = max(rd, key=rd.get)
    for k, v in rd.items():
        if k != most_common and k in original_rating:
            diff = rd[most_common]-rd[k]
            multiplier = int(diff / rd[k])
            subsample = neighbourhood.loc[neighbourhood['overall'] == k]
            if multiplier > 0:
                to_append = pd.concat([subsample]*multiplier)
                neighbourhood = pd.concat([neighbourhood, to_append])

    # represent the full dataframe
    full_df = pd.concat([neighbourhood, test_target])

    ## output the new rating distribution after data balncing if desired
    # rd = dict(neighbourhood["overall"].value_counts())
    # o_rd = dict(full_df["overall"].value_counts())
    # print("Neighbourhood Rating distribution: ", rd)
    # print("Overall Rating distribution: ", o_rd)
    # num_reviews = full_df[full_df.columns[0]].count()
    # print("Number of reviews: ", num_reviews)

    # the number of features in each method for adjusted r**2 metric
    #1 LightGBM
    #2 RandomForest
    #3 6-SVD
    #4 6-SVD++
    #5 SVD
    #6 NN
    #7 LightGBM (NonPers)
    feature_nums = {1: 6, 2: 6, 3: 6, 4: 6, 5: 1, 6: 6, 7: 1}

    # select executing a model or conducting analysis, and then select which models to evaluate and how
    valid2 = False
    while not valid2:
        if model_analysis is not None:
            choice = model_analysis
        else:
            choice = input("Model or Analysis? [M/A]: ")
        if choice.upper() == "M":
            model_analysis = choice
            valid2 = True
            valid = False
            while not valid:
                if pers_yes is not None:
                    yn = pers_yes
                else:
                    yn = input("Evaluate 1 Model [Y] or All? [Y/A]: ")
                    # yn = "Y"   # The evaluation of non-personality models is now conducted by choosing to evaluate on
                    #             # all methods
                if yn.upper() == "Y":
                    pers_yes = yn
                    print("Using Personality....")
                    valid = True
                    valid_in = False
                    if method_choice is None:
                        print("[L] - LightGBM")
                        print("[R] - Random Forest")
                        print("[S] - FFM-SVD")
                        print("[P] - FFM-SVD++")
                        print("[N] - Neural Network")
                    while not valid_in:
                        while not valid_in:
                            if method_choice is not None:
                                method = method_choice
                            else:
                                method = input("Please choose a method above: ")
                            if method.upper() == "L":
                                method_choice = method
                                valid_in = True
                                recommendations_df = create_lightgbm(full_df, train, test, chosen_user, "L", code, True, g_test_split, True)
                                m_name = "LightGBM"
                                p_type = True
                                b_type = True
                                m_choice = 1
                            elif method.upper() == "R":
                                method_choice = method
                                valid_in = True
                                recommendations_df = create_lightgbm(full_df, train, test, chosen_user, "R", code, True, g_test_split, True)
                                m_name = "RandomForest"
                                m_choice = 2
                                p_type = True
                                b_type = True
                            elif method.upper() == "S":
                                method_choice = method
                                valid_in = True
                                recommendations_df, dp_result = approach1(full_df, train, test, chosen_user, False, code, True, dp_round, True, g_test_split)
                                dp_round = dp_result
                                m_name = "6-SVD"
                                m_choice = 3
                                p_type = True
                                b_type = True
                            elif method.upper() == "P":
                                method_choice = method
                                valid_in = True
                                recommendations_df, dp_result = approach1(full_df, train, test, chosen_user, True, code, True, dp_round, True, g_test_split)
                                dp_round = dp_result
                                m_name = "6-SVD++"
                                p_type = True
                                b_type = True
                                m_choice = 4
                            elif method.upper() == "N":
                                method_choice = method
                                valid_in = True
                                m_choice = 6
                                recommendations_df = baseline_nn(full_df, train, test, chosen_user, code, True, g_test_split, 5, True)
                                m_name = "NeuralNet"
                                p_type = True
                                b_type = True

                # Run all models
                elif yn.upper() == "A":
                    valid = True
                    pers_yes = yn
                    g_all = True
                    results = []

                    # allow for the testing for SVD bucketing
                    if dp_round is not None:
                        dp = dp_round
                    else:
                        if g_test_bucket is None:
                            inp = input("Test SVD Bucketing? [Y/N]: ")
                            if inp.upper() == "N":
                                valid_dp = False
                                while not valid_dp:
                                    try:
                                        dp = int(input("Round SVD by (Recommended: 5): "))
                                        dp_round = dp
                                        valid_dp = True
                                    except:
                                        print("Invalid - Please enter an integer")
                            g_test_bucket = inp

                    # Execute all models
                    print("LightGBM...")
                    results.append(
                        ["LightGBM", True, False, create_lightgbm(full_df, train, test, chosen_user, "L", code, False, g_test_split, True), 1])
                    results.append(
                        ["LightGBM", False, False,
                         create_lightgbm(full_df, train, test, chosen_user, "L", code, False, g_test_split, False), 7])
                    print("Personality Random Forest...")
                    results.append(
                        ["RandomForest", True, False, create_lightgbm(full_df, train, test, chosen_user, "R", code, False, g_test_split, True), 2])
                    results.append(
                        ["RandomForest", False, False, create_lightgbm(full_df, train, test, chosen_user, "R", code, False, g_test_split, False),
                         2])
                    print("FFM-SVD...")
                    if g_test_bucket.upper() == "N":
                        results.append(["6-SVD", True, False, approach1(full_df, train, test, chosen_user, False, code, False, dp, True, g_test_split)[0], 3])
                    else:
                        maxdp = 10
                        for i in range(1, maxdp+1):
                            name = "6-SVD-" + str(i)
                            dp = i
                            results.append([name, True, False,
                                            approach1(full_df, train, test, chosen_user, False, code, False, dp, True,
                                                      g_test_split)[0], 3])
                    print("Non-Personality SVD...")
                    results.append(["SVD", False, False, approach1(full_df, train, test, chosen_user, False, code, False, dp, False, g_test_split)[0], 5])
                    print("FFM-SVD++...")
                    results.append(
                        ["6-SVD++", True, False, approach1(full_df, train, test, chosen_user, True, code, False, dp, True, g_test_split)[0], 4])
                    print("Non-Personality SVD++...")
                    results.append(["SVD++", False, False, approach1(full_df, train, test, chosen_user, True, code, False, dp, False, g_test_split)[0], 5])
                    print("Baseline NeuralNet...")
                    n_epoch = 5
                    results.append(
                        ["Baseline NeuralNet", True, False, baseline_nn(full_df, train, test, chosen_user, code, True, g_test_split, n_epoch, True), 6])
                    results.append(
                        ["Baseline NeuralNet", False, False, baseline_nn(full_df, train, test, chosen_user, code, True, g_test_split, n_epoch, False), 6])
                else:
                    print("Invalid input, please enter a 'Y' or an 'N'")

            # Format combined results
            if yn.upper() == "A":
                df_dict = defaultdict(list)

                print()
                # evaluate responses and prepare for outputting
                for result in results:
                    response = evaluate(code, result[0], result[1], result[2], result[3], train, test, chosen_user, False, feature_nums[result[4]], full_df)
                    df_dict["Model"].append(result[0])
                    df_dict["Personality"].append(result[1])
                    df_dict["Ratings-balanced"].append(result[2])
                    df_dict["RMSE 1"].append(response[0])
                    df_dict["RMSE 2"].append(response[1])
                    df_dict["RMSE 3"].append(response[2])
                    df_dict["RMSE 4"].append(response[3])
                    df_dict["RMSE 5"].append(response[4])
                    df_dict["Overall RMSE"].append(response[5])
                    df_dict["MAE"].append(response[7])
                    df_dict["Adjusted R2"].append(response[6])
                    df_dict["Prediction StD"].append(response[8])
                    df_dict["Predictions Made"].append(response[9])
                result_df = pd.DataFrame(df_dict)
                formatted_df = result_df.copy()
                formatted_df = formatted_df.set_index("Model")

                g_results.append(result_df)

                # save results
                if code == "M":
                    prefix = "Movies"
                elif code == "D":
                    prefix = "Music"
                elif code == "K":
                    prefix = "Kindle"
                elif code == "V":
                    prefix = "Video_Games"
                elif code == "P":
                    prefix = "Pet_Supplies"
                elif code == "G":
                    prefix = "Patio_Lawn_Garden"
                elif code == "S":
                    prefix = "Sports_and_Outdoors"
                elif code == "C":
                    prefix = "CDs_and_Vinyl"
                formatted_df.to_csv("saved_results/" + prefix + "_results.csv")
                print(formatted_df)

                resultant_dfs = []
                for result in results:
                    resultant_dfs.append([result[0], result[3], result[1], result[2]])

                # evalute approaches across users
                global_eval(result_df, resultant_dfs, test, chosen_user, full_df)

                print()

            # output the most recommended items
            else:
                print()
                print("Most recommended")
                print(recommendations_df.head(10))
                print()

                response = evaluate(code, m_name, p_type, b_type, recommendations_df, train, test, chosen_user, True, feature_nums[m_choice], full_df)
                g_results.append(response)

        # conduct exploratory analysis
        elif choice.upper() == "A":
            model_analysis = choice
            exploratory_analysis(full_df, user_num, code)
            valid2 = True
        else:
            print("Invalid input")


# function to control the outputting of results
def output_results():
    global g_results
    global g_all

    if not g_all:
        df_dict = defaultdict(list)

        for result in g_results:
            df_dict["RMSE 1"].append(result[0])
            df_dict["RMSE 2"].append(result[1])
            df_dict["RMSE 3"].append(result[2])
            df_dict["RMSE 4"].append(result[3])
            df_dict["RMSE 5"].append(result[4])
            df_dict["Overall RMSE"].append(result[5])
            df_dict["MAE"].append(result[7])
            df_dict["Adjusted R2"].append(result[6])
            df_dict["Prediction StD"].append(result[8])
        result_df = pd.DataFrame(df_dict)
        formatted_df = result_df.copy()
        print(formatted_df)

    else:
        combined_results = pd.concat(g_results)
        combined_saved = combined_results.copy()
        tallies = {}
        for index, row in combined_results.iterrows():
            if row["Model"] not in tallies:
                tallies[row["Model"]] = row["Predictions Made"]
            else:
                tallies[row["Model"]] += row["Predictions Made"]

        weighted_avg_dict = {}
        for index, row in combined_results.iterrows():
            key = row["Model"] + "-" + str(row["Personality"]) + "-" + str(row["Ratings-balanced"])
            if key not in weighted_avg_dict:
                weighted_avg_dict[key] = [row.tolist()]
            else:
                temp = weighted_avg_dict[key]
                temp.append(row.tolist())
                weighted_avg_dict[key] = temp

        weighted_avg_df = pd.DataFrame(columns=combined_results.columns)
        weighted_avg_df = weighted_avg_df.drop(columns=["RMSE 1", "RMSE 2", "RMSE 3", "RMSE 4", "RMSE 5"])

        wms = []
        # take weighted average of all columns
        for i in range(3, 12):
            temp = []
            for k, v in weighted_avg_dict.items():
                # v is a list of results (so a list of list)
                weighted_mean = []
                # for each result, add the target item the number of times that the prediction has been made
                for result in v:
                    for count in range(result[-1]):
                        weighted_mean.append(result[i])
                wm = sum(weighted_mean) / len(weighted_mean)
                temp.append(wm)
            wms.append(temp)

        # get tallies
        counts = []
        for k, v in weighted_avg_dict.items():
            sum_total = 0
            for result in v:
                sum_total += result[-1]
            counts.append(sum_total)


        new_row = {
            'Model': [v[0][0] for k, v in weighted_avg_dict.items()],
            'Personality': [v[0][1] for k, v in weighted_avg_dict.items()],
            'Ratings-balanced': [v[0][2] for k, v in weighted_avg_dict.items()],
            # removed these as stdev now does the job and also averaging identical user-item reviews means there are
            #       are more ratings categories than just 1-5 ints
            # "RMSE 1": wms[0],
            # "RMSE 2": wms[1],
            # "RMSE 3": wms[2],
            # "RMSE 4": wms[3],
            # "RMSE 5": wms[4],
            "Overall RMSE": wms[5],
            "MAE": wms[6],
            "Adjusted R2": wms[7],
            "Prediction StD": wms[8],
            "Predictions Made": counts
        }
        new_row_df = pd.DataFrame(new_row)
        weighted_avg_df = pd.concat([weighted_avg_df, new_row_df])
        print(weighted_avg_df)

main()