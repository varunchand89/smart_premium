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



class label_1:
    def __init__(self, cato_train):
        self.cato_train = cato_train
        self.encoder = LabelEncoder()  # Initialize the encoder

    def label(self):  # Pass the dataframe dynamically
        self.encoder.fit(self.cato_train['Location'])  # Fit on training data
        encoder_col = self.encoder.transform(self.cato_train['Location'])  # Transform new data
        loc_df = pd.DataFrame(encoder_col, columns=['Location_encoder'])
        cato_df_1 = pd.concat([self.cato_train, loc_df], axis=1)
        cato_df_1 = cato_df_1.drop("Location", axis=1)
        return cato_df_1