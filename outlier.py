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


class out_1:
    def __init__(self,cato_train_1):
       self.cato_train_1 = cato_train_1

    def Find_outlier(self):
      outliers_dict = {}
      for column in self.cato_train_1.columns[1:]:
         q1 = self.cato_train_1[column].quantile(0.25)
         q3 = self.cato_train_1[column].quantile(0.75)
         iqr = q3 - q1

         lowerbound = q1 - 1.5*iqr
         upperbound = q3 + 1.5*iqr

         outliers = [x for x in self.cato_train_1[column] if x < lowerbound or x > upperbound]
         outliers_dict[column] = outliers
      outliers = {k : v for k,v in outliers_dict.items() if v}
      return outliers.keys()