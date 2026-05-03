import pandas as pd
import os

data_folder = "../data/raw"

files = [
    "depression_2019_features_tfidf_256.csv",
    "anxiety_2019_features_tfidf_256.csv",
    "bipolarreddit_2019_features_tfidf_256.csv",
    "bpd_2019_features_tfidf_256.csv",
    "schizophrenia_2019_features_tfidf_256.csv",
]

data = []

for file in files:

    path = os.path.join(data_folder, file)

    df = pd.read_csv(path)

    df = df.sample(600)

    label = file.split("_")[0]

    df["label"] = label

    data.append(df)

dataset = pd.concat(data)

dataset.to_csv("../data/final_dataset_small.csv",index=False)

print("Small dataset created")