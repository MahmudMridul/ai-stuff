import pandas as pd
# import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import seaborn as sb
import matplotlib.pyplot as plt

# print(os.getcwd())

data = pd.read_csv("datasets/customer_sentiment.csv")


plt.figure(figsize=(10, 6))
sb.countplot(data=data, x='age_group', hue='sentiment', palette=['red', 'gray', 'green'])
plt.title('Sentiment Distribution by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Sentiment', title_fontsize=11)
plt.tight_layout()
# plt.show()
plt.savefig("age_sentiment.png")
plt.close()

# features = data.drop(["customer_id", "sentiment", "gender"], axis=1)
# target = data["sentiment"]


# print(features.columns)
# print(data.head(20))

