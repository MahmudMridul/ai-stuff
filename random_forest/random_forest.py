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

data_set = pd.read_csv("datasets/play_tennis.csv")

features = data_set.drop(['Day', 'PlayTennis'], axis=1)
label = data_set['PlayTennis']

feature_encoders = {}

for column in features.columns:
    feature_encoder = LabelEncoder()
    features[column] = feature_encoder.fit_transform(features[column])
    feature_encoders[column] = feature_encoder

target_encoder = LabelEncoder()
label = target_encoder.fit_transform(label)

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)


model = RandomForestClassifier(
    n_estimators=20, 
    criterion='gini', 
    max_depth=1,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=3,
    bootstrap=True,
    verbose=1,
    max_samples=10)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
# print(x_test)
# print(target_encoder.inverse_transform(y_test))
# print(target_encoder.inverse_transform(y_pred))
print(accuracy)

