from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

FILE_NAME = "../../data/streams_2016_1023_sampled_users.csv"
SAVE_NAME = "../../data/diversity_data.csv"


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
    # Entries with a z-score above the threshold will be dropped (also instant-skips of listening_time < X)
    raw.drop(raw[z > z_threshold].index, inplace=True)
    raw.drop(raw[raw.listening_time < min_listen_time].index, inplace=True)
    return raw[(raw["origin"] != "unknown")]


def get_songs_data(cleaned_data, recommended=True, obs_threshold=7):
    """
    Gets the number of songs and plays for each user (calculates S/P diversity index)
    :param obs_threshold: user must have this number of observations of more to be kept in the dataset
    :param cleaned_data: data after removing outliers
    :param recommended: gets the recommended part of the dataset if True, the organic part otherwise
    :return: a pandas Series with [user_id, S/P] as its rows
    """
    # Separating based on entry's origin
    if recommended:
        songs_data = cleaned_data[cleaned_data["origin"].str.contains("flow|reco")]
    else:
        songs_data = cleaned_data[~cleaned_data["origin"].str.contains("flow|reco")]

    # Computing for each user every S and their respective P
    songs_data = songs_data.groupby(["user_id", "sng_id"]).size()

    # Determine and drop cases with too few observations (see obs_threshold)
    thresh_tester = songs_data.sum(level="user_id")
    thresh_tester = thresh_tester[thresh_tester < obs_threshold]
    songs_data.drop(thresh_tester.index.values, inplace=True)

    # No exception will be raised even on empty series (in case no value is above obs-threshold)
    songs_data = songs_data.groupby(["user_id"]).size().divide(songs_data.sum(level="user_id"))
    return songs_data


@DeprecationWarning
def compute_old_diversity(user_songs, threshold=7):
    """
    Calculates diversity of with a dict of songs and their respective plays. DEPRECATED since 06.03.2020
    Diversity used is S/P, which stands for SONGS / PLAYS (0 < S/P < 1)
    :param user_songs: dict of songs listened to by users. {user1_id: {song_1: amount_1, ...}, ...}
    :param threshold: users must have this amount of observations in the dataset or more to be kept
    :return: {user1_id: diversity_ratio, ...}
    """
    to_delete = []

    for user, listened in user_songs["user_id"]:
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
    # users = np.unique(raw_data.user_id)
    users = raw_data.user_id.unique()
    user_nb = users.shape[0]
    print(f"Individual users: {user_nb}")

    print("Computing diversity ratio...")
    rec_data = get_songs_data(raw_data, recommended=True)
    org_data = get_songs_data(raw_data, recommended=False)

    print("Organizing data...")
    # get_songs_data produces a pd.Series, which we turn into a DataFrame before adjusting some column names
    org_data = org_data.to_frame()
    rec_data = rec_data.to_frame()
    org_data.index.names = ["user"]
    rec_data.index.names = ["user"]
    org_data.columns = ["ratio"]
    rec_data.columns = ["ratio"]
    org_data["recommended"] = 0
    rec_data["recommended"] = 1
    org_data = pd.concat([org_data, rec_data])

    org_data.to_csv(SAVE_NAME)
