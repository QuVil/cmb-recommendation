import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

from formatting.data_prep import clean_data, is_reco


def bin_artists(data):
    """
    takes in raw data from streams (with added art_id -- see "formatting/add_artists.py") and bins artists together
    according to their "mainstream" status -- i.e. how much they are listened to.
    :param data: DataFrame ["user_id", "ts", "sng_id", "album_id", "listening_time", "listen_type", "origin", "art_id"]
    :return: the same DataFrame with added columns [..., "bin_nb", "recommended"]
    """
    print("Counting artist occurences...")
    artist_count = raw_data["art_id"].value_counts(normalize=False).sort_values(ascending=False).to_frame() \
        .reset_index().rename(columns={"index": "art_id", "art_id": "plays"})

    print("Binning artists...")
    # Cutting into 100 bins (duplicate bins allowed -- and necessary for small datasets)
    artist_count["bin"] = pd.qcut(artist_count["plays"], 100, duplicates="drop")
    # Assigning corresponding numbers to bins
    artist_count["bin_nb"] = (artist_count["bin"].cat.codes + 1).values[::-1]

    data = data.merge(artist_count[["art_id", "bin_nb"]], left_on="art_id", right_on="art_id")

    # Adding 'recommended' column
    data["recommended"] = data["origin"].map(is_reco)

    return data


if __name__ == "__main__":
    FILE_NAME = "../../data/artist_streams.csv"
    SAVE_NAME = "../../data/mainstream_data.csv"

    sns.set_palette(sns.color_palette("colorblind"))

    print("Reading data...")
    raw_data = pd.read_csv(FILE_NAME)
    raw_data = clean_data(raw_data, z_threshold=3)
    print(f"Data shape: {raw_data.shape[0]} lines, {raw_data.shape[1]} columns")

    # Bin artists together
    artist_data = bin_artists(raw_data)

    # TODO normalisation en fonction d'un delta? La normalisation se fait pour l'instant par ratio
    # Occurences of music plays in each bin
    bin_distrib = artist_data.groupby("bin_nb").size()
    # Normalized
    bin_distrib /= bin_distrib.sum()

    print("Analyzing users...")
    main_dist = artist_data.groupby(["user_id", "recommended", "bin_nb"]).size()

    test_users = np.random.choice(artist_data["user_id"].unique(), 6, replace=False)
    recommended_choices = [True, False]
    aggregate = pd.DataFrame(columns=["user_id", "recommended", "bin_nb", "ratio"])

    # plt.style.use("ggplot")
    # fig, ax = plt.subplots(1, figsize=(8, 6))
    # fig.suptitle("Mainstream ratios")
    for user, reco in itertools.product(test_users, recommended_choices):
        try:
            # to_frame() may raise an error on account of the data being size 1 (an int64 cannot be framed)
            user_dist = main_dist[user][reco].to_frame()
            user_dist.columns = ["ratio"]

            # Normalizing with regards to "average" consumption
            user_dist["ratio"] /= bin_distrib * user_dist["ratio"].sum()

            # Adding data to total frame for later seaborn plotting (useless for quick-and-dirty mpl)
            user_dist = user_dist.reset_index()
            user_dist["user_id"] = user
            user_dist["recommended"] = reco
            aggregate = pd.concat([aggregate, user_dist], ignore_index=True)
        except AttributeError as e:
            warnings.warn(f"User data of size 1 ignored for user {user}, recommended {reco}")
            pass

        # Mpl plotting -- plotting can be done either by these two lines and the plt.legend or the above total framing
        # ax.plot(user_dist.index, user_dist["ratio"], "-", label=f"{user[:5]} (" + ("rec" if reco else "org") + ")")
        # plt.fill_between(user_dist.index, user_dist["ratio"])
    # plt.legend(loc="upper left", title="", frameon=False)

    sns.relplot(x="bin_nb", y="ratio", hue="recommended", style="recommended", kind="line", col="user_id", col_wrap=3,
                dashes=False, markers=True, estimator=None, data=aggregate, legend="brief")
    sns.despine()
    plt.show()

    # sns.distplot(artist_count["bin_nb"])
    # plt.show()

    # print(artist_count.tail())
    # sns.relplot(x="index", y="count", kind="line", data=artist_count.reset_index())
    # plt.hist(artist_count["count"], cumulative=True, density=True, bins=100)
    # plt.show()

    # print("Fitting function to distribution of play counts...")
    #
    # def func(x, a, b, c):
    #     return a * np.exp(-b * x) + c
    #
    # popt, pcov = curve_fit(func, np.arange(1, len(artist_count.index)+1),
    #                        artist_count["count"].values, p0=[10e-3 for i in range(3)])
    # plt.plot(np.arange(1, len(artist_count.index)+1), func(np.arange(1, len(artist_count.index)+1), *popt),
    #          "r-", label="Fitted Curve")
    # plt.show()
