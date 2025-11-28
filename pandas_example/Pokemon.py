import re

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.options.mode.copy_on_write = True

filePath = "datasets/pokemon_data.csv"

data = pd.read_csv(filePath, header=0)
data = data.drop(["#"], axis=1)
print(data.head(10))

# get all the columns as list
# columns = data.columns.tolist()
# print(columns)

# get values of a column
names = data["name"]
print(names)

# get values of a column and range of rows
# first_ten_names = data["Name"][0:10]

# get values of multiple column
# general_info = data[["Name", "HP", "Attack", "Defense", "Speed"]]

# get values of multiple column and range of rows
# first_ten_info = data[["Name", "HP", "Attack", "Defense", "Speed"]][0:10]

# get all info of a row
# charizard = data.iloc[6]

# get all info of range of rows
# charizard_family = data.iloc[4:7]

# get cell value [row, column]
# charizard_attack = data.iloc[6, 4]

# get range of rows and range of columns
# charizard_basic_info = data.iloc[4:7, 0:6]

# add a column
# charizard_basic_info["HP"] = charizard_basic_info["HP"] * 2

# get rows satisfying specific conditions
# fire_flying_types = data.loc[(data["Type 1"] == "Fire") & (data["Type 2"] == "Flying")]

# get specific rows satisfying specific conditions
# fire_types_info = data.loc[
#     (data["Type 1"] == "Fire") & (data["Type 2"] == "Flying"),
#     ["Name", "HP", "Attack", "Defense"],
# ]

# fire_type = data.loc[(data["Type 1"] == "Fire")]

# sort values
# fire_type_sorted = fire_type.sort_values("Name", ascending=True, inplace=False)

# sort values with by multiple colums
# fire_type_sorted = fire_type.sort_values(["Name", "HP"], ascending=[True, False])

# add a column
# fire_type["Rating"] = (
#     (fire_type["HP"] + fire_type["Attack"] + fire_type["Defense"] + fire_type["Speed"])
#     / 1000
# ) * 100
# fire_type["Total"] = fire_type.iloc[:, 3:10].sum(axis=1)

# delete a column
# fire_type = fire_type.drop(columns=["Generation"])

# save data frame as file
# fire_type.to_csv("FireTypePokemon.csv", index=False)
# fire_type.to_excel("FireTypePokemon.xlsx", index=False)

# reset index
# fire_type = fire_type.reset_index()
# fire_type = fire_type.reset_index(drop=True)

# filter with regex
# grass_water_type = data.loc[
#     data["Type 1"].str.contains("grass|water", flags=re.IGNORECASE, regex=True)
# ]

# get distinct values of a column
# unique_types = data["Type 1"].unique().tolist()
