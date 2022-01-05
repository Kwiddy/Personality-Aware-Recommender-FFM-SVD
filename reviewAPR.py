from empath import Empath
import pandas as pd
from os.path import exists
from tqdm import tqdm
import liwc
import re
from collections import Counter
from sklearn import preprocessing
import numpy as np

# NEED TO CHANGE SO THAT IT OPERATES ON EVERY USER

# See LIWC_FFM_CORRELATIONS.xlsx for references
liwc_to_ffm = {
    "pronoun": [0.06, 0.06, -0.21, 0.11, -0.02],
    "i": [0.12, 0.01, -0.16, 0.05, 0],
    "we": [-0.07, 0.11, -0.1, 0.18, 0.03],
    "you": [-0.15, 0.16, -0.12, 0.08, 0],
    "shehe": [0.02, 0.04, -0.06, 0.08, -0.08],
    "they": [0.02, 0.04, -0.06, 0.08, -0.08],
    "article": [-0.11, -0.04, 0.2, 0.03, 0.09],
    "past": [0.03, -0.01, -0.16, 0.1, 0],
    "present": [0.06, -0.01, -0.16, 0, -0.06],
    "future": [-0.02, -0.06, -0.08, -0.01, -0.01],
    "prep": [-0.04, -0.04, 0.17, 0.07, 0.06],
    "negate": [0.11, -0.05, -0.13, -0.03, -0.17],
    "number": [-0.07, -0.12, -0.08, 0.11, 0.04],
    "swear": [0.11, 0.06, 0.06, -0.21, -0.14],
    "social": [-0.06, 0.15, -0.14, 0.13, -0.04],
    "family": [-0.07, 0.09, -0.17, 0.19, 0.05],
    "friend": [-0.08, 0.15, -0.01, 0.11, 0.06],
    "human": [-0.05, 0.13, -0.09, 0.07, -0.12],
    "affect": [0.07, 0.09, -0.12, 0.06, -0.06],
    "posemo": [-0.02, 0.1, -0.15, 0.18, 0.04],
    "negemo": [0.16, 0.04, 0, -0.15, -0.18],
    "anx": [0.17, -0.03, -0.02, -0.03, -0.05],
    "anger": [0.13, 0.03, 0.03, -0.23, -0.19],
    "sad": [0.1, 0.02, -0.03, 0.01, -0.11],
    "cogmech": [0.13, -0.06, -0.09, -0.05, -0.11],
    "insight": [0.08, 0, -0.08, 0.01, -0.05],
    "cause": [0.11, -0.09, -0.02, -0.11, -0.12],
    "discrep": [0.13, -0.07, -0.12, -0.04, -0.13],
    "tentat": [0.12, -0.11, -0.06, -0.07, -0.1],
    "certain": [0.13, 0.1, -0.06, 0.05, -0.1],
    "inhib": [0.09, -0.13, -0.07, -0.08, -0.05],
    "incl": [-0.02, 0.09, 0.11, 0.18, 0.07],
    "excl": [0.1, -0.06, 0, -0.07, -0.16],
    "percept": [0.05, 0.09, -0.11, 0.05, -0.1],
    "see": [-0.01, 0.03, -0.04, 0.09, 0.01],
    "hear": [0.02, 0.12, -0.08, 0.01, -0.12],
    "feel": [0.1, 0.06, -0.01, 0.1, -0.05],
    "body": [0.02, 0.1, -0.04, 0.09, -0.07],
    "sexual": [0.03, 0.17, 0, 0.08, -0.06],
    "ingest": [-0.01, 0.08, -0.15, 0.03, -0.04],
    "motion": [-0.02, 0.02, -0.22, 0.14, 0.04],
    "space": [-0.09, 0.02, -0.11, 0.16, 0.04],
    "time": [0.01, -0.02, -0.22, 0.12, 0.09],
    "work": [0.07, -0.08, 0.04, -0.07, 0.07],
    "achieve": [0.01, -0.09, -0.05, 0.05, 0.14],
    "leisure": [-0.05, 0.08, -0.17, 0.15, 0.06],
    "home": [0, 0.03, -0.2, 0.19, 0.05],
    "money": [0.04, -0.04, -0.04, -0.11, -0.08],
    "relig": [-0.03, 0.11, 0.05, 0.06, -0.04],
    "death": [0.03, 0.11, 0.15, -0.13, -0.12],
    "assent": [0.05, 0.07, -0.11, 0.02, -0.09]
}


def calc_empath(corpus):
    lexicon = Empath()
    emph_scores = lexicon.analyze(corpus, normalize=True)
    sorted_emph_scores = {k: v for k, v in sorted(emph_scores.items(), key=lambda item: item[1])}
    
    
    print(sorted_emph_scores)
    for k, v in sorted_emph_scores.items():
        if v != 0.0:
            print(k + ": " + str(v))
    
    return sorted_emph_scores


# This function is derived from https://github.com/chbrown/liwc-python
# may want to use a smarter tokenizer
def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


# This function is derived from https://github.com/chbrown/liwc-python
# THIS IS ONLY LIWC2007, NEEDS TO BE UPDATED IF POSSIBLE
def calc_liwc(corpus):
    # parse, category_names = liwc.load_token_parser('./LIWC/LIWC2007_English100131.dic')
    parse, category_names = liwc.load_token_parser('./LIWC/LIWC2015_Dictionary.dic')
    tokens = tokenize(corpus)
    counts = Counter(category for token in tokens for category in parse(token))
    return counts


def convert_to_ffm(liwc):
    category_scores = []
    for k, v in liwc.items():
        if k in liwc_to_ffm:
            to_add = [contribution * v for contribution in liwc_to_ffm[k]]
            category_scores.append(to_add)

    # get total category score
    ffm_scores = []
    for i in range(5):
        total = 0
        for j in range(len(category_scores)):
            total += category_scores[j][i]
        ffm_scores.append(round(total, 3))

    # print(ffm_scores)

    # need to turn into the right scale, or just normalize?

    # normalize
    np_ffm_scores = np.asarray(ffm_scores)
    # print(np_ffm_scores)
    norm_ffm_scores = preprocessing.normalize([np_ffm_scores])
    # print(norm_ffm_scores)
    final_ffm_scores = norm_ffm_scores.tolist()

    # need to have normalized scores because different users will have different amounts of data
    #   

    # print(final_ffm_scores[0])
    return final_ffm_scores[0]


# Create scores for each user
def generate_scores(df, parent_path, ext):

    # Get all unique user IDs
    ID_list = df['reviewerID'].tolist()

    data = []
    
    print("Generating scores...")
    for i in tqdm(range(len(ID_list))):
        reviewerID = ID_list[i]
        entry = df.loc[df['reviewerID'] == reviewerID]["corpus"]
        new = entry.to_frame()
        corpus = new.iloc[0]['corpus']
        corpus = corpus.lower()

        # emph_scores = calc_empath(corpus)

        liwc_scores = calc_liwc(corpus)
        ffm_scores = convert_to_ffm(liwc_scores)

        # save to dataframe
        data.append([reviewerID, ffm_scores[0], ffm_scores[1], ffm_scores[2], ffm_scores[3], ffm_scores[4]])
    
    ffm_df = pd.DataFrame(data, columns=['reviewerID', 'Extroversion', "Agreeableness", "conscientiousness", "Neurotisicm", "Openness_to_Experience"])
    new_path = parent_path + ext + "_personality.csv" #"Movie_and_TV_5_personality.csv"
    ffm_df.to_csv(new_path)

    return ffm_df


# combine user reviews
def create_corpora(df, parent_path, ext):
    new_path = parent_path + ext + "_reviews.csv"# "Movie_and_TV_5_reviews.csv"

    print("Creating reviews dataframe...")
    if exists(new_path):
        reviews_df = pd.read_csv(new_path)
    else:
        reviews_df = df[["reviewerID", "reviewText"]].copy()
        reviews_df = reviews_df.dropna()
        reviews_df.to_csv(new_path)
        
    # create corpora df
    new_path = parent_path + ext + "_corpora.csv" #+ "Movie_and_TV_5_corpora.csv"
    print("Creating corpus dataframe...")
    if exists(new_path):
        corpora_df = pd.read_csv(new_path)
    else:
        corpora = {}

        for index, row in tqdm(reviews_df.iterrows()):
            if row["reviewerID"] in corpora:
                try:
                    corpora[row["reviewerID"]] += " " + row["reviewText"]
                except:
                    print("DATA ERROR ON ROW BELOW")
                    print(row)
            else:
                corpora[row["reviewerID"]] = row["reviewText"]

        print("Number of reviewers: " + str(len(corpora)))
        avg=[]
        for key, value in corpora.items():
            avg.append(len(value))
        mean = sum(avg) / len(avg)
        print("Average Corpus length " + str(mean))

        corpora_df = pd.DataFrame.from_dict(corpora, orient="index").reset_index()
        corpora_df.columns = ["reviewerID", "corpus"]

        # print("Printing Corpora DF")
        # print(corpora_df)
        print("Saving Corpora DF to CSV")
        corpora_df.to_csv(new_path)

    return corpora_df


def review_APR(df, parent_path, extension):
    # "Digital_Music_5.csv"
    sub_extension = extension[:-4]

    # create corpus for user
    corpora_df = create_corpora(df, parent_path, sub_extension)

    # create scores from corpus
    new_path = parent_path + sub_extension + "_personality.csv"# + "Movie_and_TV_5_personality.csv"
    if exists(new_path):
        print("Personality scores already exist...")
        ffm_df = pd.read_csv(new_path)
    else:
        ffm_df = generate_scores(corpora_df, parent_path, sub_extension)

    return ffm_df