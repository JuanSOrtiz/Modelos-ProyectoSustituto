# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:39:45 2024

@author: JUAN
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.compose import make_column_selector
from loguru import logger
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--test_data_file', required=True, type=str, help='a csv file with test data')
parser.add_argument('--model_file', required=True, type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if sets overwrites the model file if it exists')

args = parser.parse_args()

model_file = args.model_file
train_data_filee  = args.train_data_file
test_data_filee  = args.test_data_file
overwrite = args.overwrite_model

if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"overwriting existing model file {model_file}")
    else:
        logger.info(f"model file {model_file} exists. exiting. use --overwrite_model option")
        exit(-1)

logger.info("loading train data")
train_data_file = pd.read_csv(train_data_filee, index_col="id")
test_data_file = pd.read_csv(test_data_filee, index_col="id")

train_data_file.isnull().sum()
train_data_file.describe()
train_data_file.drop_duplicates(inplace=True, keep="first")

preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object)),
    ("scale", StandardScaler(), make_column_selector(dtype_include=np.number)),
])

X_train = train_data_file.drop("NObeyesdad",axis=1)
y_train = train_data_file["NObeyesdad"]

preprocess.fit(pd.concat([X_train,test_data_file]))
X_train = preprocess.transform(X_train)
X_test = preprocess.transform(test_data_file)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

X_train1, X_test1, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



xgb1 = xgb.XGBClassifier(objective='multi:softmax', num_class=7, random_state=42)
xgb1.fit(X_train1, y_train)


X_test1_df=pd.DataFrame(X_test1)
X_test1_df.to_csv("test_data_input.csv", index=False)

y_test_df=pd.DataFrame(y_test)
y_test_df.to_csv("test_data_target.csv", index=False)

logger.info(f"saving model to {model_file}")
with open(model_file, "wb") as f:
    pickle.dump(xgb1, f)