# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:12:44 2024

@author: hp
"""
import pandas as pd 

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , StandardScaler , MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#------------------------------------------------------------------------------------------------------

def impute_data(df, col, strategy): 
    imputer = SimpleImputer(strategy=strategy)
    df[col] = imputer.fit_transform(df[[col]])
    return df

#------------------------------------------------------------------------------------------------------

def drop_na(df, col):
    df = df.dropna(subset=[col])
    return df
#------------------------------------------------------------------------------------------------------



def encode_data(df, col_encode, encode_way):
    if encode_way == "Label Encode":
        label_encoder = LabelEncoder()
        df[col_encode] = label_encoder.fit_transform(df[col_encode])
    elif encode_way == "OneHot Encode":
        encoder = OneHotEncoder()
        encoded_labels = encoder.fit_transform(df[[col_encode]]).toarray()
        encoded_columns = encoder.get_feature_names_out([col_encode])
        encoded_df = pd.DataFrame(encoded_labels, columns=encoded_columns)
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=[col_encode], inplace=True)
    return df


#------------------------------------------------------------------------------------------------------

def scale_data(df, col, method):
    x = df.drop(columns=[col])
    if method == "minmax":
        scaler = MinMaxScaler()
        scaled_x = scaler.fit_transform(x)
    elif method == "standard":
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(scaled_x, columns=x.columns)
    df = pd.concat([scaled_df, df[col]], axis=1)
    return df 

#------------------------------------------------------------------------------------------------------

def PCA_data(df, col, n):
    x = df.drop(columns=[col])
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n)])
    df = pd.concat([principal_df, df[col]], axis=1)
    return df