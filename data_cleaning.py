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


class cleaning:
    def __init__(self,dx):
       self.dx = dx


    def data_cleaning(self):
      
      self.dx['Age'] = self.dx['Age'].fillna(self.dx['Age'].mean().astype(int))
      self.dx['Annual Income'] = self.dx['Annual Income'].fillna(self.dx['Annual Income'].mean().astype(int))
      self.dx['Marital Status'] = self.dx['Marital Status'].fillna("Single")
      self.dx['Number of Dependents'] = self.dx['Number of Dependents'].fillna(self.dx['Number of Dependents'].mean().astype(int))
      self.dx['Occupation'] = self.dx['Occupation'].fillna("Employed")
      self.dx['Health Score'] = self.dx['Health Score'].fillna(self.dx['Health Score'].mean().astype(int))
      self.dx['Previous Claims'] = self.dx['Previous Claims'].fillna(self.dx['Previous Claims'].mean().astype(int))
      self.dx['Credit Score'] = self.dx['Credit Score'].fillna(self.dx['Credit Score'].mean().astype(int))
      self.dx['Customer Feedback'] =self.dx['Customer Feedback'].fillna("Average")
      self.dx['Insurance Duration'] = self.dx['Insurance Duration'].fillna("9")
      self.dx['Vehicle Age']= self.dx['Vehicle Age'].fillna(self.dx['Vehicle Age'].mean().astype(int))
      self.dx['Insurance Duration'] = self.dx['Insurance Duration'].astype(int)

      return self.dx.iloc[:,[1,2,3,5,8,9,11,13,14,17,20]]
    
