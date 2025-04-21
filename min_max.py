import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np 
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from cato import cato_1



class mima:
    def __init__(self,cato_train_1,categorical_columns_cato_train_2):
        self.cato_train_1 = cato_train_1
        self.categorical_columns_cato_train_2 = categorical_columns_cato_train_2



    def min_max(self):
        minmax = MinMaxScaler()
        max_value = minmax.fit_transform(self.cato_train_1[self.categorical_columns_cato_train_2])
        max_value_frame = pd.DataFrame(max_value,columns =self.categorical_columns_cato_train_2 )
        cato_df_mer_1 = pd.concat([self.cato_train_1,max_value_frame],axis =1)
        duplicate_cols = cato_df_mer_1.columns.duplicated(keep='last')
        cato_df_mer_1 = cato_df_mer_1.loc[:, ~duplicate_cols]
        return cato_df_mer_1