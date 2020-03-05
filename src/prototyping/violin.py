import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Reading data...")

tips = sns.load_dataset("tips")
print(tips.dtypes)
ax = sns.violinplot(x="day", y="total_bill", hue="sex",
                    data=tips, palette="Set2", split=True,
                    scale="count")
plt.show()
