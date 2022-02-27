import math
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def evaluate(df_code, model_name, personality_type, balance_type, results, train, test, user, display, num_features):
    # get the test split for the chosen user
    user_test = test.loc[test['reviewerID'] == user]
    user_test_relev = user_test[["asin", "overall"]].copy()

    # rename overall to actual
    user_test_relev = user_test_relev.rename(columns={"overall": "actual"})

    # set asin as index for both
    user_test_relev = user_test_relev.set_index('asin')
    results = results.set_index('asin')

    comparison = user_test_relev.join(results)

    result = calc_metrics(df_code, comparison, display, num_features, model_name, personality_type, balance_type)

    return result


def calc_metrics(df_code, df, disp, k, model_name, personality_type, balance_type):
    if disp:
        results_graph(df, model_name, personality_type, balance_type, df_code)
    rmse_df = df.copy()
    rmse_df["RMSE"] = (df["actual"]-df["predictions"])**2
    rmse_df["AbsError"] = abs(df["actual"]-df["predictions"])
    
    rmse = math.sqrt(rmse_df["RMSE"].mean())

    # get the standard deviation of the predictions
    std = float(df["predictions"].std())

    mae = rmse_df["AbsError"].mean()

    if disp:
        print(rmse_df)
        print()
        print(rmse_df.describe())
        print()
        print("RMSE: ", rmse)
        print("MAE: ", mae)
        print("StD: ", std)
        print()

    # RMSE for each different scoring
    rmse_list = [[], [], [], [], []]
    for index, row in rmse_df.iterrows():
        rmse_list[int(row["actual"])-1].append(float(row["RMSE"]))

    r2 = r2_score(df["actual"], df["predictions"])
    n = df[df.columns[0]].count() # number of samples
    # k = number of independant variables:
    ar2 = 1 - ((1-r2)*(n-1)/(n-k-1))

    if disp:
        print("1 Rating RMSE: ", math.sqrt(sum(rmse_list[0]) / len(rmse_list[0])))
        print("2 Rating RMSE: ", math.sqrt(sum(rmse_list[1]) / len(rmse_list[1])))
        print("3 Rating RMSE: ", math.sqrt(sum(rmse_list[2]) / len(rmse_list[2])))
        print("4 Rating RMSE: ", math.sqrt(sum(rmse_list[3]) / len(rmse_list[3])))
        print("5 Rating RMSE: ", math.sqrt(sum(rmse_list[4]) / len(rmse_list[4])))
        print()
        print("Adjusted R2 Score: ", ar2)
        print()

    # [1 RMSE, 2 RMSE, 3 RMSE, 4 RMSE, 5 RMSE, rmse]
    return [math.sqrt(sum(rmse_list[0]) / len(rmse_list[0])), math.sqrt(sum(rmse_list[1]) / len(rmse_list[1])),
            math.sqrt(sum(rmse_list[2]) / len(rmse_list[2])), math.sqrt(sum(rmse_list[3]) / len(rmse_list[3])),
            math.sqrt(sum(rmse_list[4]) / len(rmse_list[4])), rmse, ar2, mae, std]


def results_graph(df, model_name, personality_type, balance_type, df_code):
    # create predictions/actual scatter plot
    title = df_code + "_" + model_name
    if personality_type:
        title += "_personality"
    else:
        title += "_no_personality"
    if balance_type:
        title += "ratings_balanced"
    else:
        title += "not_rating_balanced"
    # plot = df.plot.scatter(x='actual', y='predictions', c='DarkBlue')
    # fig = plot.get_figure()
    # fig.savefig("saved_results/" + title + ".png")

    scatplot = sns.regplot(x=df['actual'], y=df['predictions'])
    fig = scatplot.get_figure()
    fig.savefig("saved_results/" + title + ".png")


def global_eval(df, indiv_dfs, test, user):

    # only keep the best of each approach
    best_idx = df.groupby(['Model'])['Overall RMSE'].transform(min) == df['Overall RMSE']
    best_df = df[best_idx].copy()

    # create rmse graph
    totals = {}
    for index, row in best_df.iterrows():
        totals[row["Model"]] = [row["RMSE 1"], row["RMSE 2"], row["RMSE 3"], row["RMSE 4"], row["RMSE 5"]]

    labels = [k for k, v in totals.items()]
    rmse1_scores = []
    rmse2_scores = []
    rmse3_scores = []
    rmse4_scores = []
    rmse5_scores = []
    for k, v in totals.items():
        rmse1_scores.append(v[0])
        rmse2_scores.append(v[1])
        rmse3_scores.append(v[2])
        rmse4_scores.append(v[3])
        rmse5_scores.append(v[4])

    X_axis = np.arange(len(labels))

    f, ax = plt.subplots(figsize=(18, 5))  # set the size that you'd like (width, height)
    plt.bar(X_axis - 0.3, rmse1_scores, 0.15, label='RMSE 1')
    plt.bar(X_axis - 0.15, rmse2_scores, 0.15, label='RMSE 2')
    plt.bar(X_axis, rmse3_scores, 0.15, label='RMSE 3')
    plt.bar(X_axis + 0.15, rmse4_scores, 0.15, label='RMSE 4')
    plt.bar(X_axis + 0.3, rmse5_scores, 0.15, label='RMSE 5')

    plt.xticks(X_axis, labels)
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model RMSE Scores per Rating")
    plt.legend()
    plt.savefig("saved_results/RMSEResults.png")

    plt.close()

    groups = [["LightGBM", "RandomForest"], ["3-SVD", "SVD"], ["3-SVD++", "SVD++"], ["Baseline NeuralNet"]]
    for group in groups:
        fig, ax = plt.subplots(figsize=(8, 8))
        for item in indiv_dfs:
            lbl = item[0]
            if lbl in group:
                if item[2]:
                    lbl += " - Personality"
                if item[3]:
                    lbl += " - Ratings-Balanced"
                resultant_df = item[1]
                user_test = test.loc[test['reviewerID'] == user]
                user_test_relev = user_test[["asin", "overall"]].copy()

                # rename overall to actual
                user_test_relev = user_test_relev.rename(columns={"overall": "actual"})

                # set asin as index for both
                user_test_relev = user_test_relev.set_index('asin')
                results = resultant_df.set_index('asin')

                comparison = user_test_relev.join(results)
                sns.regplot(x='actual', y='predictions', data=comparison, ax=ax, label=lbl)

        ax.set(ylabel='Prediction', xlabel='Actual')
        ax.legend()
        if "LightGBM" in group:
            title = "saved_results/Predictions_True_Scatter_Tree.png"
        elif "SVD" in group:
            title = "saved_results/Predictions_True_Scatter_SVD.png"
        elif "SVD++" in group:
            title = "saved_results/Predictions_True_Scatter_SVDpp.png"
        else:
            title = "saved_results/Predictions_True_Scatter_NN.png"
        plt.savefig(title)
        plt.show()
        plt.clf()
        plt.close()