import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
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
from lable import label_1
from outlier import out_1
from min_max import mima
from data_cleaning import cleaning


st.title("Smart premium")

with st.form(key="data_form"):

  Age = st.number_input("Enter Your age")
  Number_of_Dependents = st.number_input("Enter Number of Dependents")
  Health_Score = st.number_input("Enter your Health Score")
  Credit_Score = st.number_input("Enter your Credit Score")
  Insurance_Duration = st.number_input("Enter your Insurance Duration")
  Gender = st.selectbox("Enter your Gender", ['Gender','Female', 'Male'])
  Smoking_Status = st.selectbox("Enter your Smoking Status", ['Smoking','No', 'Yes'])
  Location = st.selectbox("Enter your Location", ['Location','Urban','Rural','Suburban'])
  Annual_Income = st.number_input("Enter your Annual Income")
  Previous_Claims = st.number_input("Enter your Previous Claims")
  submitted = st.form_submit_button("Submit")

if submitted:
    data = {"Age" : [Age],"Number of Dependents":[Number_of_Dependents],"Health Score":[Health_Score],"Credit Score":[Credit_Score],"Insurance Duration":[Insurance_Duration],"Gender":[Gender],"Smoking Status":[Smoking_Status],"Location":[Location],"Annual Income":[Annual_Income],"Previous Claims":[Previous_Claims]}
    data_1 = pd.DataFrame(data)
    st.success("Wait for your premium amount")
    dx = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/train.csv')
    categorical_columns = ['Gender','Smoking Status']

    train_data_1 = cleaning(dx)
    train_data = train_data_1.data_cleaning()


    cato_3 = train_data

    test_data = data_1

    cato_test_10  = cato_1(data_1,categorical_columns)
    cato_test = cato_test_10.categorical()

    
    cato_test_101  = label_1(cato_test)
    cato_test_11 = cato_test_101.label()

    cato_test_1 = cato_test_11

    cato_test_1 = cato_test_1[sorted(cato_test_1.columns)]
    # Get the MLflow client
    #client = mlflow.tracking.MlflowClient()
    model_name = "XGR_Boost" 
        
    scaler = MinMaxScaler()
    col_1 =['Annual Income','Previous Claims','Predicted_Target_premium _account']
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = "models:/XGR_Boost/1"  # Replace 'latest' with a specific version if needed
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    predicted_values = loaded_model.predict(cato_test_1)
    cato_test_1["Predicted_Target_premium_account"] = predicted_values

    le = LabelEncoder()
    
    # Fit label encoder using original training data categories
    le.fit(cato_3['Location'])
    
    # Apply inverse transform
    cato_test_1['Location_encoder'] = le.inverse_transform(cato_test_1['Location_encoder'])

    col_2 = ['Gender','Smoking Status']
    col_3 = ['Gender_Female','Gender_Male','Smoking Status_No','Smoking Status_Yes']
    one = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one.fit(cato_3[col_2])
    cato_test_1[col_2] = one.inverse_transform(cato_test_1[col_3])
    cato_test_3 = cato_test_1.drop(cato_test_1.columns[[6,7,8,9]], axis=1)


    scaler.fit(cato_3[['Premium Amount']])
    cato_test_3['Predicted_Target_premium_account'] = scaler.inverse_transform(cato_test_3[['Predicted_Target_premium_account']])
    
st.write(cato_test_3['Predicted_Target_premium_account'])
    
    
   
    
    

