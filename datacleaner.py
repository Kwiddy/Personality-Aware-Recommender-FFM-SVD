import pandas as pd


# find instances where a user has rated multiple items and remove, replacing with the mean
def clean(path, new_path):
    df = pd.read_csv(path)

    kept = df[["reviewerID", "asin", "overall"]]
    grouped = kept.groupby(["reviewerID", "asin"]).mean().reset_index()

    joined = pd.merge(df, grouped, on=["reviewerID", "asin"])

    joined = joined.set_index(["reviewerID", "asin"])
    reduced = joined[~joined.index.duplicated(keep='first')]

    reduced = reduced.drop(columns=["overall_x"])
    reduced = reduced.rename(columns={"overall_y": "overall"}).reset_index()
    print(reduced.columns)
    reduced = reduced.drop(columns=["Unnamed: 0"])

    reduced.to_csv(new_path)