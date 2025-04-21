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





class cato_1:

    def __init__(self,cato_3,categorical_columns):
        self.cato_3 = cato_3
        self.categorical_columns = categorical_columns

    def categorical(self):
      encoder = OneHotEncoder(sparse_output=False,categories=[['Female', 'Male'],['Yes','No']])
      one_hot_encoded = encoder.fit_transform(self.cato_3[self.categorical_columns])
      manual_column_names = ['Gender_Female', 'Gender_Male', 'Smoking Status_Yes', 'Smoking Status_No']
      one_df = pd.DataFrame(one_hot_encoded,columns = manual_column_names)
      cato_df = pd.concat([self.cato_3,one_df],axis =1)
      cato_df = cato_df.drop((['Gender','Smoking Status']),axis =1)
      return cato_df
    
