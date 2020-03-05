from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

FILE_NAME = "../../data/streams_2016_1023_sampled_users.csv"
SAVE_NAME = "../../data/testing.csv"


def is_reco(origin):
    return origin[:3] == "rec" or origin == "flow"


def clean_data(raw, z_threshold=2.5, min_listen_time=20):
    """
    Drops entries with abnormal listening_time (z-score too high or time too low).
    Also removes entries with unknown origin
    :param min_listen_time: entries where the user has listened for less than this (s) are dropped
    :param z_threshold: maximum z-score allowed in the dataset
    :param raw: raw data to be processed
    :return: clean dataset (pd.DataFrame)
    """
    print("Computing z-score...")
    z = np.abs(stats.zscore(raw["listening_time"]))

    print("Cleaning...")
    # Entries with a z-score above the threshold will be dropped (also instant-skips)
    raw.drop(raw[z > z_threshold].index, inplace=True)
    raw.drop(raw[raw.listening_time < min_listen_time].index, inplace=True)
    return raw[(raw["origin"] != "unknown")]


def compute_diversity(user_songs, threshold=7):
    """
    Diversity used is S/P, which stands for SONGS / PLAYS (always 0 < S/P < 1)
    :param user_songs: dict of songs listened to by users. {user1_id: {song_1: amount_1, ...}, ...}
    :param threshold: users must have this amount of observations in the dataset or more to be kept
    :return: {user1_id: diversity_ratio, ...}
    """
    to_delete = []

    for user, listened in user_songs.items():
        # Avoid division by zero for empty cases and keep a record of users with nb_observations < threshold
        if (plays := sum(list(listened.values()))) > threshold:
            user_songs[user] = len(listened) / plays
        else:
            to_delete.append(user)

    # Delete empty entries
    for user in to_delete:
        del user_songs[user]
    return user_songs


if __name__ == "__main__":
    print("Reading data...")
    raw_data = pd.read_csv(FILE_NAME)
    raw_data = clean_data(raw_data)

    print(f"Data shape: {raw_data.shape[0]} lines, {raw_data.shape[1]} columns")

    print("Preparing data...")
    users = np.unique(raw_data.user_id)
    user_nb = users.shape[0]
    print(f"Individual users: {user_nb}")

    # songs = np.unique(raw_data.sng_id)
    # song_nb = songs.shape[0]
    # print(f"Individual songs: {song_nb}")

    # Separate recommended and organic listening
    data_org = dict(zip(users, [defaultdict(int) for i in range(user_nb)]))
    data_rec = dict(zip(users, [defaultdict(int) for j in range(user_nb)]))

    print("Organizing data...")
    entry_nb = raw_data.shape[0]
    # Create a dict with a list of songs heard by users and the amount of plays for each song
    for idx, entry in raw_data.iterrows():
        if is_reco(entry.origin):
            data_rec.get(entry.user_id)[entry.sng_id] += 1
        else:
            data_org.get(entry.user_id)[entry.sng_id] += 1

        # Note that the progress is incorrect if the data is cleaned up (indices on the dataframe are not updated)
        if idx % 5000 == 0:
            print(f"{idx}/{entry_nb} ({((idx / entry_nb) * 100):9.2f} %) done")

    print("Calculating diversity ratio...")
    data_org = compute_diversity(data_org)
    data_rec = compute_diversity(data_rec)

    data = list(zip(data_org.keys(), data_org.values(), [0 for i in range(len(data_org))]))
    data.extend(list(zip(data_rec.keys(), data_rec.values(), [1 for i in range(len(data_rec))])))

    dataframe = pd.DataFrame(data, columns=["user", "ratio", "recommended"])
    dataframe["recommended"] = dataframe["recommended"].astype('category')
    # print(dataframe)

    dataframe.to_csv(SAVE_NAME, index=False)
