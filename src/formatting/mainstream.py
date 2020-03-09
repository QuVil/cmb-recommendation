import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

from formatting.data_prep import clean_data

if __name__ == "__main__":
    FILE_NAME = "../../data/artist_streams.csv"
    SAVE_NAME = "../../data/mainstream_data.csv"

    print("Reading data...")
    raw_data = pd.read_csv(FILE_NAME)
    raw_data = clean_data(raw_data, z_threshold=3)
    print(f"Data shape: {raw_data.shape[0]} lines, {raw_data.shape[1]} columns")

    print("Counting artist occurences...")
    artist_count = raw_data['art_id'].value_counts(normalize=False).sort_values(ascending=False).to_frame()\
        .reset_index().rename(columns={'index': 'art_id', 'art_id': 'plays'})

    print("Binning artists...")
    # Cutting into 100 bins (duplicates allowed)
    artist_count["bin"] = pd.qcut(artist_count["plays"], 100, duplicates="drop")
    # Assigning numbers to corresponding bins
    artist_count["bin_nb"] = (artist_count["bin"].cat.codes + 1).values[::-1]

    raw_data = raw_data.merge(artist_count[["art_id", "bin_nb"]], left_on="art_id", right_on="art_id")

    # TODO normalisation en fonction du delta
    # Occurences of music plays in each bin
    bin_distrib = raw_data.groupby("bin_nb").size()
    # Normalized
    bin_distrib /= bin_distrib.sum()

    # Analysis for users
    main_dist = raw_data.groupby(["user_id", "bin_nb"]).size()

    test_users = np.random.choice(raw_data["user_id"].unique(), 3, replace=False)
    for user in test_users:
        user_dist = main_dist[user].to_frame()
        user_dist.columns = ["amount"]

        # Normalizing with regards to "average" consumption
        user_dist["amount"] /= bin_distrib * user_dist["amount"].sum()

        plt.plot(user_dist.index, user_dist["amount"], 'o-')

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
    #          'r-', label="Fitted Curve")
    # plt.show()
