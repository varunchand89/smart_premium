import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlflow.models import infer_signature
from cato import cato_1
from lable import label_1
import pickle
import mlflow
import mlflow.sklearn
import numpy as np 


st.title("Smart premium")

with st.form(key="data_form"):

  Age = st.number_input("Enter Your age",min_value=0, max_value=100, value=0,key="Age")
  Number_of_Dependents = st.number_input("Enter Number of Dependents",min_value=0, max_value=10)
  Health_Score = st.number_input("Enter your Health Score")
  Credit_Score = st.number_input("Enter your Credit Score")
  Insurance_Duration = st.number_input("Enter your Insurance Duration",min_value=0, max_value=10)
  Gender = st.selectbox("Enter your Gender", ['Gender','Female', 'Male'])
  Smoking_Status = st.selectbox("Enter your Smoking Status", ['Smoking','No', 'Yes'])
  Location = st.selectbox("Enter your Location", ['Location','Urban','Rural','Suburban'])
  Annual_Income = st.number_input("Enter your Annual Income",min_value=0, max_value=1000000)
  Previous_Claims = st.number_input("Enter your Previous Claims")
  submitted = st.form_submit_button("Submit")

if submitted:
    data = {"Age" : [Age],"Number of Dependents":[Number_of_Dependents],"Health Score":[Health_Score],"Credit Score":[Credit_Score],"Insurance Duration":[Insurance_Duration],"Gender":[Gender],"Smoking Status":[Smoking_Status],"Location":[Location],"Annual Income":[Annual_Income],"Previous Claims":[Previous_Claims]}
    data_1 = pd.DataFrame(data)
    st.success("Wait for your premium amount")
    
    categorical_columns = ['Gender','Smoking Status']
    test_data = data_1

    cato_test_10  = cato_1(data_1,categorical_columns)
    cato_test = cato_test_10.categorical()

    
    cato_test_101  = label_1(cato_test)
    cato_test_11 = cato_test_101.label()

    cato_test_1 = cato_test_11

    with open("XGR_model_1.pkl", "rb") as f:
       loaded_model = pickle.load(f)
    
    XGR_Boost = loaded_model
   
        
       
    cato_test_1 = cato_test_1[sorted(cato_test_1.columns)]
    # Get the MLflow client
        #client = mlflow.tracking.MlflowClient()
        
        
    scaler = MinMaxScaler()
    col_1 =['Annual Income','Previous Claims','Predicted_Target_premium_account']


    
    predicted_values = XGR_Boost.predict(cato_test_1)
    cato_test_1["Predicted_Target_premium_account"] = predicted_values
    st.write(np.expm1(cato_test_1["Predicted_Target_premium_account"]))
    




