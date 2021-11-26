from empath import Empath
import pandas as pd
from os.path import exists
from tqdm import tqdm

# NEED TO CHANGE SO THAT IT OPERATES ON EVERY USER

lexicon = Empath()
# print(lexicon.analyze("he hit the other person", normalize=True))

# Create scores for each user
def generate_scores():
    print("hi")

# combine user reviews
def create_corpora(df, parent_path):
    new_path = parent_path + "Movie_and_TV_5_reviews.csv"

    print("Creating reviews dataframe...")
    if exists(new_path):
        reviews_df = pd.read_csv(new_path)
    else:
        reviews_df = df[["reviewerID", "reviewText"]].copy()
        reviews_df = reviews_df.dropna()
        reviews_df.to_csv(new_path)
        
    # create corpora df
    new_path = parent_path + "Movie_and_TV_5_corpora.csv"
    print("Creating corpus dataframe")
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

        print("Printing Corpora DF")
        print(corpora_df)
        print("Saving Corpora DF to CSV")
        corpora_df.to_csv(new_path)


def review_APR(df, parent_path):
    # create corpus for user
    create_corpora(df, parent_path)

    # create scores from corpus