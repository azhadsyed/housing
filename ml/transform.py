"""
transform.py is responsible for connecting to the database of Craigslist listings,
loading the listings into memory, and transforming the raw data into a human-
readable but model-friendly csv file that can be used to train a model.

Here are some definitions that apply to this project:

1. Human-readable: if you gave this data to a human statistician they could conduct
meaningful analysis
2. Model friendly: If you used this data as an model input, you should be 
able meaningfully train a model from it.
3. Machine friendly: If you gave this data to a python-implementation of a 
machine learning algorithm, it would be able to then predict on future data with-
out throwing an error.

future goals - if one part of the transform job fails, cache the data as is and 
load it into the next runtime. On a successful run, delete the cache

"""

from pymongo import MongoClient
import pandas as pd, numpy as np
import re, json

# Step 0: the Mongo Connection string should have a default, but be CLI-driven
from config import connection_string

# Step 1: Connect to the database (Mongo)
client = MongoClient(connection_string)
db = client["housing"]
collection = db["listings"]
listings = list(collection.find())

# Step 2: Read the data from Mongo into memory (filter for training/extra columns, no garbage)
dataframe = pd.DataFrame(listings)

# Step 3: Drop garbage columns (columns not needed for either cleaning or training)
garbage = [
    "_id",
    "address",
    "attrs",
    "available",
    "created",
    "datetime",
    "deleted",
    # "geotag",
    # "id",
    "images",
    "has_image",
    "last_updated",
    "repost_of",
    "url",
    "where",
    "area",  # rip, build the front end before coming back to this
]

dataframe.drop(garbage, axis=1, inplace=True)

# 4a. dropping rooms for rent
def search_for_substring(text, pattern):
    return True if re.search(pattern, text, flags=re.IGNORECASE) else False


pattern = r"room[s]* for rent"
in_name = dataframe["name"].apply(search_for_substring, pattern=pattern)
in_body = dataframe["body"].apply(search_for_substring, pattern=pattern)
dataframe = dataframe[(in_name == False) & (in_body == False)]  # this could be cleaner

# 4b. cleaning Price, bedrooms and bathrooms
dataframe["price"] = dataframe["price"].apply(lambda x: float(re.sub(r"[,$]+", "", x)))
dataframe = dataframe[dataframe.price.between(10, 10000)]

keywords = ["shared", "split"]
dataframe.loc[dataframe.bathrooms.apply(lambda x: x in keywords), "bathrooms"] = 0.5

dataframe["bathrooms"] = pd.to_numeric(dataframe.bathrooms, errors="coerce")
dataframe["bedrooms"] = pd.to_numeric(dataframe.bedrooms, errors="coerce")

# 4c. In columns containing only true and np.nan, switch nan to False
hashable = [i for i in dataframe.columns if list not in set(dataframe[i].apply(type))]
to_fill = [i for i in hashable if set(dataframe[i].unique()) == {True, np.nan}]
value = {i: False for i in to_fill}
dataframe.fillna(value, inplace=True)

# 4d. In columns with categorical values, impute the mode
categorical_features = [i for i in hashable if dataframe[i].dtype == "object"]
[categorical_features.remove(i) for i in ["name", "body", "id"]]

categorical_modes = {
    i: dataframe[i].value_counts().index[0] for i in categorical_features
}

dataframe.fillna(categorical_modes, inplace=True)

# 4e. For bedrooms and bathrooms, search for missing values in 'name' and 'body' fields
# this is the dirtiest part of the codebase
def numberwords_to_digits(string):
    """ receives a string, returns the string with 'number' words converted to digits"""
    numberwords = dict(
        one="1",
        two="2",
        three="3",
        four="4",
        five="5",
        six="6",
        seven="7",
        eight="8",
        nine="9",
        ten="10",
        zero="0",
    )
    string = string.lower()
    for k, v in numberwords.items():
        string = string.replace(k, v)
    return string


def find_groups(text, pattern):
    r = re.search(text, pattern)
    return max(r.groups()) if r else np.nan


def conditional_mutate(row, missing_field, source_field, pattern):
    if pd.isnull(row[missing_field]):
        text = numberwords_to_digits(row[source_field])
        return find_groups(text, pattern)
    else:
        return row[missing_field]


bedroom_expression = r"(?<!\$)(\d+\.?[\d]*)[\s\-]*(?:bd|br|room|bed|bdrm|bedroom|cuarto)s*(?!\w)|bed(?:room)*s*: (\d+\.*\d*)|(\d)B\dB"
bathroom_expression = r"(\d+\.?[\d]*)[\s\-]*(?:ba|bath|bth|bathroom|bano)s*(?!\w)|bath(?:room)*s*: (\d+\.*\d*)|\dB(\d)B"


def impute_rooms(room_name, pattern):
    dataframe[room_name] = dataframe.apply(
        conditional_mutate,
        missing_field=room_name,
        source_field="name",
        pattern=pattern,
        axis=1,
    )

    dataframe[room_name] = dataframe.apply(
        conditional_mutate,
        missing_field=room_name,
        source_field="body",
        pattern=pattern,
        axis=1,
    )


impute_rooms("bedrooms", bedroom_expression)
impute_rooms("bathrooms", bathroom_expression)

# unpack lat and long
dataframe["latitude"] = dataframe["geotag"].apply(lambda x: x[0] if x else None)
dataframe["longitude"] = dataframe["geotag"].apply(lambda x: x[1] if x else None)

# Step 5: drop extra columns (that were needed for cleaning, not for training)
extra = ["name", "body", "geotag"]

dataframe.dropna(inplace=True)
dataframe.drop(extra, axis=1, inplace=True)

# Step 6: transform the training fields into a csv file
dataframe.to_csv(open("data/data.csv", "w"), index=False)

# Step 7: cache the options from categorical variables to options.json, as well
# as the column order excluding ID and price

column_order = list(dataframe.columns)
del column_order[:2]

options = {
    "laundry": sorted(list(dataframe.laundry.unique())),
    "parking": sorted(list(dataframe.parking.unique())),
    "housing type": sorted(list(dataframe.housing_type.unique())),
    "column order": column_order,
    "categorical features": categorical_features,
}
with open("data/options.json", "w") as f:
    json.dump(options, f)
