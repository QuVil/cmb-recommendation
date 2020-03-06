import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = "../../data/diversity_data.csv"

print("Reading data...")
data = pd.read_csv(FILE_NAME)

# Dirty workaround -- should try and figure out how to use "hue" and split better
data["test"] = 1

print("Plotting violin graph...")
# sns.violinplot(x="test", y="ratio", hue="recommended", data=data, split=True, scale="count", bw=0.12)
sns.catplot(x="recommended", y="ratio", data=data)
plt.show()
