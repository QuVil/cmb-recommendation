import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("Reading data...")
data = pd.read_csv("../../data/streams_2016_1023_sampled_users.csv")

print(f"individual users: {np.unique(data.user_id).shape[0]}")
print(f"Listening times are between {np.min(data.listening_time)} and {np.max(data.listening_time)}")

print(np.unique(data.origin))

sns.kdeplot(data.listening_time, shade=True)
plt.show()
