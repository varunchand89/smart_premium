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
from lable import label_1
from outlier import out_1
from min_max import mima
from data_cleaning import cleaning


st.title("Smart premium")

with st.form(key="data_form"):

  Age = st.text_input("Enter Your age")
  Number_of_Dependents = st.text_input("Enter Number of Dependents")
  Health_Score = st.text_input("Enter your Health Score")
  Credit_Score = st.text_input("Enter your Credit Score")
  Insurance_Duration = st.text_input("Enter your Insurance Duration")
  Gender = st.selectbox("Enter your Gender", ['Gender','Female', 'Male'])
  Smoking_Status = st.selectbox("Enter your Smoking Status", ['Smoking','No', 'Yes'])
  Location = st.selectbox("Enter your Location", ['Location','Urban','Rural','Suburban'])
  Annual_Income = st.text_input("Enter your Annual Income")
  Previous_Claims = st.text_input("Enter your Previous Claims")
  submitted = st.form_submit_button("Submit")

if submitted:
    data = {"Age" : [Age],"Number of Dependents":[Number_of_Dependents],"Health Score":[Health_Score],"Credit Score":[Credit_Score],"Insurance Duration":[Insurance_Duration],"Gender":[Gender],"Smoking Status":[Smoking_Status],"Location":[Location],"Annual Income":[Annual_Income],"Previous Claims":[Previous_Claims]}
    data_1 = pd.DataFrame(data)
    st.success("Wait for your premium amount")
    
    
    
    
    dx = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/train.csv')
    dz = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/test.csv')
    
    mlflow.set_experiment("Random_forest_Regressor_Automation_6")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name="Minmax"):
       
        train_data_1 = cleaning(dx)
        train_data = train_data_1.data_cleaning()


        cato_3 = train_data

        categorical_columns = ['Gender','Smoking Status']
        
        cato_train_99 = cato_1(cato_3,categorical_columns)
        cato_train = cato_train_99.categorical()

    
        cato_train_101 = label_1(cato_train)
        cato_train_1 = cato_train_101.label()

    
        train_out_90 = out_1(cato_train_1)
        train_out = train_out_90.Find_outlier()

    
        categorical_columns_cato_train_2=[]
        for cato_train_1_1 in train_out:
           categorical_columns_cato_train_2.append(cato_train_1_1)
            
        cato_train_38 = mima(cato_train_1,categorical_columns_cato_train_2)
        cato_train_3 = cato_train_38.min_max()

        A = cato_train_3.drop('Premium Amount', axis=1)
        b = cato_train_3['Premium Amount']
        A = A[sorted(A.columns)]

        test_data = data_1

        cato_test_10  = cato_1(data_1,categorical_columns)
        cato_test = cato_test_10.categorical()

    
        cato_test_101  = label_1(cato_test)
        cato_test_1 = cato_test_101.label()

    
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2 , random_state = 42)
        


    
        regression_tree_1 = RandomForestRegressor(n_estimators=10, random_state=42,oob_score=False,max_depth=5,n_jobs=-1,max_features="sqrt")
        regression_tree_1.fit(A_train,b_train)

        B_pred = regression_tree_1.predict(A_test)
        signature = infer_signature(A_train, regression_tree_1.predict(A_train))
        
        mae = mean_absolute_error(b_test,B_pred)
        mse = mean_squared_error(b_test, B_pred)
        rmse = np.sqrt(mean_squared_error(b_test, B_pred))
        r2 = r2_score(b_test, B_pred)
        name = "Minmax"
        model = regression_tree_1
    
        mlflow.log_param("Scaler", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2 Score", r2)
        mlflow.sklearn.log_model(
        sk_model=regression_tree_1,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="RandomForestRegressor1",
    )
        
       
        cato_test_1 = cato_test_1[sorted(cato_test_1.columns)]
    # Get the MLflow client
        #client = mlflow.tracking.MlflowClient()
        model_name = "RandomForestRegressor1" 
        
        scaler = MinMaxScaler()
        col_1 =['Annual Income','Previous Claims','Predicted_Target_premium _account']


    
        predicted_values = regression_tree_1.predict(cato_test_1)
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
        
    mlflow.end_run()


    st.write(cato_test_3['Predicted_Target_premium_account'])

