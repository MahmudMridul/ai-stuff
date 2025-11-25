import pandas as pd
# import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# print(os.getcwd())

data_set = pd.read_csv("Datasets/play_tennis.csv")

features = data_set.drop(['Day', 'PlayTennis'], axis=1)
label = data_set['PlayTennis']

label_encoders = {}

for column in features.columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

print(features)

le = LabelEncoder()
label = le.fit_transform(label)

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)


model = RandomForestClassifier(
    n_estimators=5, 
    criterion='gini', 
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=4,
    bootstrap=True,
    verbose=1,
    max_samples=6)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

