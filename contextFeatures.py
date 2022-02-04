# https://www.tensorflow.org/recommenders/examples/context_features?hl=en

import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

class MovielensModel(tfrs.models.Model):

  def __init__(self, use_contexts):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      UserModel(use_contexts),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      MovieModel(),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and context features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "user_id": features["user_id"],
        "context": features["context"],
    })
    movie_embeddings = self.candidate_model(features["movie_title"])

    return self.task(query_embeddings, movie_embeddings)

class MovieModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)

class UserModel(tf.keras.Model):

    def __init__(self, use_contexts):
        super().__init__()

        self._use_contexts = use_contexts

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])

        if use_contexts:
            self.context_embedding = tf.keras.Sequential([
                tf.keras.layers.Discretization(context_buckets.tolist()),
                tf.keras.layers.Embedding(len(context_buckets) + 1, 32),
            ])
            self.normalized_context = tf.keras.layers.Normalization(
                axis=None
            )

            self.normalized_context.adapt(context)

    def call(self, inputs):
        if not self._use_contexts:
            return self.user_embedding(inputs["user_id"])

        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.context_embedding(inputs["context"]),
            tf.reshape(self.normalized_context(inputs["context"]), (-1, 1)),
        ], axis=1)

# def advanced_model(full_df, train, chosen_user):

def getAsinCode(asin):
    global asin_convert
    return asin_convert[asin]


def getReviewerCode(id):
    global reviewer_convert
    return reviewer_convert[id]


df = pd.read_csv("reduced.csv")

asin_convert = {}
reviewer_convert = {}
id = 0
for index, row in df.iterrows():
    if row["asin"] not in asin_convert:
        asin_convert[row["asin"]] = id
        asin_convert[id] = row["asin"]
    if row["reviewerID"] not in reviewer_convert:
        reviewer_convert[row["reviewerID"]] = id
        reviewer_convert[id] = row["reviewerID"]
    id += 1


df["asin_code"] = df["asin"].apply(getAsinCode)
df["reviewer_code"] = df["reviewerID"].apply(getReviewerCode)

df = df.drop(columns=["reviewText", "reviewerName", "summary", "vote", "image", "reviewTime", "verified", "Unnamed: 0", "style", "Unnamed: 0.1", "reviewerID", "asin"])
print(df)
df_tensor = tf.convert_to_tensor(df)
df_movies = tf.convert_to_tensor(df["asin_code"])
print(df_tensor)

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

print(ratings)
print(type(ratings))
print(movies)
print(type(movies))
print("-----")
# print(ratings2)
# print(type(ratings2))
# print(movies2)
# print(type(movies2))

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    # "context": x["timestamp"]
    "context": tf.random.uniform(shape=[], minval=0, maxval=100, dtype=tf.int64)
})

movies = movies.map(lambda x: x["movie_title"])
# for item in movies:
#     print(item)

# print()cls

# for item in df_movies:
#     print(item)
# movies = df_movies.numpy()
# print(df_movies)
# print(type(df_movies))

record_defaults = [999]
dataset = tf.data.experimental.CsvDataset("reduced.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
# dataset = dataset.map(lambda x: x["rating"])

print()
print(movies)
print(type(movies))
print(dataset)
print(type(dataset))

movies = dataset

# exit()
# exit()
# movies = df_movies
# movies = df_tensor.map(lambda x: x["asin_code"])

# print(ratings)
# print(type(ratings))
# print(movies)
# print(type(movies))
#
# for data in movies:
#     print(data,type(data))
# print("------")
# for data in ratings:
#     print(data, type(data))
#
# exit()

########################

context = np.concatenate(list(ratings.map(lambda x: x["context"]).batch(100)))

max_context = context.max()
min_context = context.min()

context_buckets = np.linspace(
    min_context, max_context, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))

#########################

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

###########################

model = MovielensModel(use_contexts=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy_nocontext = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy_nocontext  = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]


###########################

model = MovielensModel(use_contexts=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]


print(f"Top-100 accuracy NO CONTEXT (train): {train_accuracy_nocontext :.2f}.")
print(f"Top-100 accuracy NO CONTEXT (test): {test_accuracy_nocontext :.2f}.")
print(f"Top-100 accuracy CONTEXT (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy CONTEXT (test): {test_accuracy:.2f}.")