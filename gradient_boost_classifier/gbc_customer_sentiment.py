import pandas as pd

# import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")

# print(os.getcwd())

data = pd.read_csv("datasets/customer_sentiment.csv")

# sentiment distribution by gender with countplot
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.countplot(data=data, x="gender", hue="sentiment", ax=ax)
# ax.set_title("Sentiment Distribution by Gender", fontsize=16, fontweight="bold", pad=20)
# ax.set_xlabel("Gender", fontsize=12)
# ax.set_ylabel("Sentiment", fontsize=12)
# plt.savefig("gender_sentiment.png")

# sentiment distribution by age group
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=data, x="age_group", hue="sentiment", ax=ax)
ax.set_title(
    "Sentiment Distribution by Age Group", fontsize=16, fontweight="bold", pad=20
)
ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("Sentiment", fontsize=12)

"""
From the countplot we see that sentimant is distributed more or less equally amonth all the genders.
So we can say that the feature gender has no influence on sentiment. But before we conclude let's
do the Chi-Square test to confirm. 
"""

plt.tight_layout()
plt.show()
plt.close()


# features = data.drop(["customer_id", "sentiment", "gender", "age_group"], axis=1)
# target = data["sentiment"]


# print(features.columns)
# print(data.head(20))
