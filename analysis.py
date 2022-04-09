import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import numpy as np
import itertools
import os
import seaborn as sns


def exploratory_analysis(df, user_num, code):
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

    domains = ["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"]

    print(personalities)
    pairs = list(itertools.combinations(domains, 2))

    print("Calculating Correlations between Personality Domains...")

    correlations = []
    for pair in pairs:
        correlation = abs(personalities[pair[0]].corr(personalities[pair[1]]))
        correlations.append([correlation, pair[0], pair[1]])

    print(correlations)

    # Print all correlations
    for item in correlations:
        print("Correlation between " + item[1] + " and " + item[2] + ": " + str(item[0]))
    print()

    # Graph the top 3 correlations
    sorted_correlations = sorted(correlations, reverse=True)
    strongest_corr = sorted_correlations[:3]

    for pair in strongest_corr:
        x = personalities[pair[1]].tolist()
        y = personalities[pair[2]].tolist()
        plt.title("Correlation between " + pair[1] + " and " + pair[2])
        plt.xlabel(pair[1])
        plt.ylabel(pair[2])
        plt.scatter(x, y)
        plt.savefig("analysis_results/" + code +"_correlation_" + pair[1] + "_" + pair[2] + ".png")
        plt.clf()

    plt.close()

    df = df.merge(personalities, on="reviewerID")
    print(df)
    # Normalize each of the personality columns -- THIS MIGHT BE WRONG
    minval = df["Extroversion"].min()
    maxval = df["Extroversion"].max()
    df["Extroversion"] = df.progress_apply(lambda x: normalize(x.Extroversion, minval, maxval), axis=1)
    minval = df["Agreeableness"].min()
    maxval = df["Agreeableness"].max()
    df["Agreeableness"] = df.progress_apply(lambda x: normalize(x.Agreeableness, minval, maxval), axis=1)
    minval = df["conscientiousness"].min()
    maxval = df["conscientiousness"].max()
    df["conscientiousness"] = df.progress_apply(lambda x: normalize(x.conscientiousness, minval, maxval), axis=1)
    minval = df["Neurotisicm"].min()
    maxval = df["Neurotisicm"].max()
    df["Neurotisicm"] = df.progress_apply(lambda x: normalize(x.Neurotisicm, minval, maxval), axis=1)
    minval = df["Openness_to_Experience"].min()
    maxval = df["Openness_to_Experience"].max()
    df["Openness_to_Experience"] = df.progress_apply(lambda x: normalize(x.Openness_to_Experience, minval, maxval), axis=1)

    print(df)


    print("--Extroversion--")
    print(df["Extroversion"].min())
    print(df["Extroversion"].max())
    print("--Agreeableness--")
    print(df["Agreeableness"].min())
    print(df["Agreeableness"].max())
    print("--conscientiousness--")
    print(df["conscientiousness"].min())
    print(df["conscientiousness"].max())
    print("--Neurotisicm--")
    print(df["Neurotisicm"].min())
    print(df["Neurotisicm"].max())
    print("--Openness_to_Experience--")
    print(df["Openness_to_Experience"].min())
    print(df["Openness_to_Experience"].max())

    print(df)

    ###############################################
    df = df.drop(columns="Unnamed: 0_y")
    corr_matrix = df.corr(method="pearson")
    sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("Pearson Correlation of Features")
    save_name = "analysis_results/" + code + "_correlationmatrix" + "_" + str(user_num) + ".png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

    ##############################################

    translate = {"Ext.": "Extroversion", "Agr.": "Agreeableness", "Con.": "conscientiousness", "Neu.": "Neurotisicm", "Ote.": "Openness_to_Experience"}

    totals = {1: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              2: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              3: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              4: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              5: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0}}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        temp_dict = totals[round(row["overall"])]
        for k, v in temp_dict.items():
            temp_dict[k] += row[translate[k]]
        totals[round(row["overall"])] = temp_dict

    print(totals.items())

    for k, v in totals.items():
        plt.bar(v.keys(), v.values())
        plt.title("Traits Appearances in Ratings of " + str(k))
        save_name = "analysis_results/" + code + "_TraitScoreCorrelation_" + str(k) + ".png"
        plt.savefig(save_name)
        plt.clf()

    plt.close()

    # normalized importance
    for k, v in totals.items():
        temp_dict = v
        minval = min(temp_dict.values())
        maxval = max(temp_dict.values())
        for k2, v2 in temp_dict.items():
            temp_dict[k2] = (v2-minval)/(maxval-minval)
        totals[k] = temp_dict

    # p0.11
    filenames = []
    for k, v in totals.items():
        plt.bar(v.keys(), v.values())
        plt.title("Normalized Traits Appearances in Ratings of " + str(k))
        save_name = "analysis_results/" + code + "_NormalizedTraitScoreCorrelation_" + str(k) + ".png"
        filenames.append(save_name)
        plt.savefig(save_name)
        plt.clf()

    with imageio.get_writer('analysis_results/' + code + '_NormalizedTraitScoreCorrelations.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    plt.close()

    #################################
    labels = ["1", "2", "3", "4", "5"]
    ext_scores = []
    agr_scores = []
    con_scores = []
    nue_scores = []
    ote_scores = []
    for k, v in totals.items():
        for k2, v2 in v.items():
            if k2 == "Ext.":
                ext_scores.append(v2)
            if k2 == "Agr.":
                agr_scores.append(v2)
            if k2 == "Con.":
                con_scores.append(v2)
            if k2 == "Neu.":
                nue_scores.append(v2)
            if k2 == "Ote.":
                ote_scores.append(v2)

    all_scores = [ext_scores, agr_scores, con_scores, nue_scores, ote_scores]
    print("ALL SCORES: ", all_scores)

    X_axis = np.arange(5)

    plt.bar(X_axis - 0.4, ext_scores, 0.15, label='Extroversion')
    plt.bar(X_axis - 0.2, agr_scores, 0.15, label='Agreeableness')
    plt.bar(X_axis, con_scores, 0.15, label='Conscientiousness')
    plt.bar(X_axis + 0.2, nue_scores, 0.15, label='Neurotisicm')
    plt.bar(X_axis + 0.4, ote_scores, 0.15, label='Openness to Experience')

    plt.xticks(X_axis, labels)
    plt.xlabel("Rating")
    plt.ylabel("Prominence")
    plt.title("Prominence of Personality Traits in Ratings Distribution")
    plt.legend()
    plt.savefig("analysis_results/" + code + "_NormalizedTraitScoreCorrelations.png")
    #################################

    print("---- Analysis Completed ----")


def normalize(value, min, max):
    return (value-min) / (max-min)