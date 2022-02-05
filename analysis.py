import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def exploratory_analysis(df):
    personalities = pd.read_csv(
        "Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Kindle_Store_5_personality.csv")

    print(personalities)

    domains = ["Extroversion", "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"]

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

    translate = {"Ext.": "Extroversion", "Agr.": "Agreeableness", "Con.": "conscientiousness", "Neu.": "Neurotisicm", "Ote.": "Openness_to_Experience"}

    totals = {1: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              2: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              3: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              4: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0},
              5: {"Ext.": 0, "Agr.": 0, "Con.": 0, "Neu.": 0, "Ote.": 0}}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        temp_dict = totals[row["overall"]]
        for k, v in temp_dict.items():
            temp_dict[k] += row[translate[k]]
        totals[row["overall"]] = temp_dict

    print(totals.items())

    for k, v in totals.items():
        plt.bar(v.keys(), v.values())
        plt.title("Traits Appearances in Ratings of " + str(k))
        save_name = "analysis_results/TraitScoreCorrelation_" + str(k) + ".png"
        plt.savefig(save_name)
        plt.clf()

    # normalized importance
    for k, v in totals.items():
        temp_dict = v
        minval = min(temp_dict.values())
        maxval = max(temp_dict.values())
        for k2, v2 in temp_dict.items():
            temp_dict[k2] = (v2-minval)/(maxval-minval)
        totals[k] = temp_dict

    for k, v in totals.items():
        plt.bar(v.keys(), v.values())
        plt.title("Normalized Traits Appearances in Ratings of " + str(k))
        plt.show()
        save_name = "analysis_results/NormalizedTraitScoreCorrelation_" + str(k) + ".png"
        plt.savefig(save_name)
        plt.clf()

def normalize(value, min, max):
    return (value-min) / (max-min)