import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from formatting.data_prep import clean_data

Z_THRESHOLD = 2.5
MIN_LISTEN_TIME = 20

print("Reading data...")
raw_data = pd.read_csv("../../data/streams_2016_1023_sampled_users.csv")

raw_lines = raw_data.shape[0]
raw_data = clean_data(raw_data, z_threshold=Z_THRESHOLD, min_listen_time=MIN_LISTEN_TIME)
print(f"Lines cleaned: {raw_lines - raw_data.shape[0]}")

print("Plotting...")
sns.distplot(raw_data["listening_time"])
plt.show()
