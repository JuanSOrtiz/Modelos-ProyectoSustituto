# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:56:18 2024

@author: JUAN
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import opendatasets as od
dataset_link="https://www.kaggle.com/competitions/playground-series-s4e2"
od.download(dataset_link)
import os
os.chdir("playground-series-s4e2 ")
os.listdir()
import pandas as pd

train_data_file=pd.read_csv('train.csv',index_col = "id")
test_data_file=pd.read_csv('test.csv',index_col = "id")

train_data_file.isnull().sum()
train_data_file.describe()
train_data_file.drop_duplicates(inplace=True , keep="first")



preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object)),
    ("scale", StandardScaler(), make_column_selector(dtype_include=np.number)),
])

X_train = train_data_file.drop("NObeyesdad",axis=1)
y_train = train_data_file["NObeyesdad"]

preprocess.fit(pd.concat([X_train,test_data_file]))
X_train = preprocess.transform(X_train)
#X_test = preprocess.transform(test_data_file)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

X_train1, X_test1, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



xgb1 = xgb.XGBClassifier(objective='multi:softmax', num_class=7, random_state=42)
xgb1.fit(X_train1, y_train)

y_pred = xgb1.predict(np.array(X_test1))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")

import pickle


X_test1_df=pd.DataFrame(X_test1)
X_test1_df.to_csv("test_data_input.csv", index=False)

y_test_df=pd.DataFrame(y_test)
y_test_df.to_csv("test_data_target.csv", index=False)



#with open("model.pkl", "wb") as f:
#    pickle.dump(xgb1, f)