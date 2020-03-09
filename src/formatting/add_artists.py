import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Reading data...")
data_songs = pd.read_json('../../data/songs.json', lines=True)
data_entries = pd.read_csv("../../data/streams_2016_1023_sampled_users.csv")
print(f"Entries before artists: {data_entries.shape[0]}")

print("Concatenating...")
data_songs = data_songs[["sng_id", "art_id"]]
data_entries = data_entries.merge(data_songs, left_on="sng_id", right_on="sng_id")
print(f"Entries after artists:  {data_entries.shape[0]}")
# test = data_entries[data_entries["art_id"].isnull()]
# test = test["sng_id"].value_counts()
# test = test.reset_index()
# print(test.columns.values)
# sns.distplot(test["sng_id"], hist=False)
# plt.show()

print("Saving...")
data_entries.to_csv("artist_streams.csv", index=False)
