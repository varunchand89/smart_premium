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
  #Marital_Status = st.text_input("Enter your Marital Status")
  #Education_Level = st.text_input("Enter your Education Level")
  #Occupation = st.text_input("Enter your Occupation")
  #Policy_Type = st.text_input("Enter your Policy Type")
  submitted = st.form_submit_button("Submit")

if submitted:
    data = {"Age" : [Age],"Number of Dependents":[Number_of_Dependents],"Health Score":[Health_Score],"Credit Score":[Credit_Score],"Insurance Duration":[Insurance_Duration],"Gender":[Gender],"Smoking Status":[Smoking_Status],"Location":[Location],"Annual Income":[Annual_Income],"Previous Claims":[Previous_Claims]}
    data_1 = pd.DataFrame(data)
    st.success("Wait for your premium amount")
    def data_cleaning(df):
      
      df['Age'] = df['Age'].fillna(df['Age'].mean().astype(int))
      df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].mean().astype(int))
      df['Marital Status'] = df['Marital Status'].fillna("Single")
      df['Number of Dependents'] = df['Number of Dependents'].fillna(df['Number of Dependents'].mean().astype(int))
      df['Occupation'] = df['Occupation'].fillna("Employed")
      df['Health Score'] = df['Health Score'].fillna(df['Health Score'].mean().astype(int))
      df['Previous Claims'] = df['Previous Claims'].fillna(df['Previous Claims'].mean().astype(int))
      df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mean().astype(int))
      df['Customer Feedback'] =df['Customer Feedback'].fillna("Average")
      df['Insurance Duration'] = df['Insurance Duration'].fillna("9")
      df['Vehicle Age']= df['Vehicle Age'].fillna(df['Vehicle Age'].mean().astype(int))
      df['Insurance Duration'] = df['Insurance Duration'].astype(int)

      return df
    def categorical(cato_3,categorical_columns):
      encoder = OneHotEncoder(sparse_output=False,categories=[['Female', 'Male'],['Yes','No']])
      one_hot_encoded = encoder.fit_transform(cato_3[categorical_columns])
      manual_column_names = ['Gender_Female', 'Gender_Male', 'Smoking Status_Yes', 'Smoking Status_No']
      one_df = pd.DataFrame(one_hot_encoded,columns = manual_column_names)
      cato_df = pd.concat([cato_3,one_df],axis =1)
      cato_df = cato_df.drop((categorical_columns),axis =1)
      return cato_df
    def label(cato_df):
      ecoder = LabelEncoder()
      encoder_col = ecoder.fit_transform(cato_df['Location'])
      loc_df = pd.DataFrame(encoder_col,columns=['Location_encoder'])
      cato_df_1 = pd.concat([cato_df,loc_df],axis = 1).drop("Location",axis = 1)
      return cato_df_1
    def Find_outlier(cato_out):
      outliers_dict = {}
      for column in cato_out.columns[1:]:
         q1 = cato_out[column].quantile(0.25)
         q3 = cato_out[column].quantile(0.75)
         iqr = q3 - q1

         lowerbound = q1 - 1.5*iqr
         upperbound = q3 + 1.5*iqr

         outliers = [x for x in cato_out[column] if x < lowerbound or x > upperbound]
         outliers_dict[column] = outliers
      outliers = {k : v for k,v in outliers_dict.items() if v}
      return outliers.keys()
    def min_max(cato_df_2,categorical_columns_cato_tst_1):
        minmax = MinMaxScaler()
        max_value = minmax.fit_transform(cato_df_2[categorical_columns_cato_tst_1])
        max_value_frame = pd.DataFrame(max_value,columns =categorical_columns_cato_tst_1 )
        cato_df_mer_1 = pd.concat([cato_df_2,max_value_frame],axis =1)
        duplicate_cols = cato_df_mer_1.columns.duplicated(keep='last')
        cato_df_mer_1 = cato_df_mer_1.loc[:, ~duplicate_cols]
        return cato_df_mer_1
    dx = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/train.csv')
    dz = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/test.csv')
    
    mlflow.set_experiment("Random_forest_Regressor_Automation_4")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name="Minmax"):
       
        train_data = data_cleaning(dx)
    
        cato_1 = train_data.iloc[:,[1,2,3,5,8,9,11,13,14,17,20]]

        categorical_columns = ['Gender','Smoking Status']
        
        cato_train = categorical(cato_1,categorical_columns)
    
        cato_train_1 = label(cato_train)
    
        train_out = Find_outlier(cato_train_1)
    
        categorical_columns_cato_train_2=[]
        for cato_train_1_1 in train_out:
          categorical_columns_cato_train_2.append(cato_train_1_1)
            
        cato_train_3 = min_max(cato_train_1,categorical_columns_cato_train_2)

        A = cato_train_3.iloc[:,:-1].values
        b = cato_train_3.iloc[:,-1].values

        test_data = data_1
        cato_test  = categorical(data_1,categorical_columns)
    
        cato_test_1  = label(cato_test)

    
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
        

    # Get the MLflow client
        #client = mlflow.tracking.MlflowClient()
        model_name = "RandomForestRegressor1" 
        
        scaler = MinMaxScaler()
        col_1 =['Annual Income','Previous Claims','Predicted_Target_premium _account']


    
        predicted_values = regression_tree_1.predict(cato_test_1)
        cato_test_1["Predicted_Target_premium_account"] = predicted_values

        le = LabelEncoder()
    
    # Fit label encoder using original training data categories
        le.fit(cato_1['Location'])
    
    # Apply inverse transform
        cato_test_1['Location_encoder'] = le.inverse_transform(cato_test_1['Location_encoder'])

        col_2 = ['Gender','Smoking Status']
        col_3 = ['Gender_Female','Gender_Male','Smoking Status_No','Smoking Status_Yes']
        one = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one.fit(cato_1[col_2])
        cato_test_1[col_2] = one.inverse_transform(cato_test_1[col_3])
        cato_test_3 = cato_test_1.drop(cato_test_1.columns[[6,7,8,9]], axis=1)


        scaler.fit(cato_1[['Premium Amount']])
        cato_test_3['Predicted_Target_premium_account'] = scaler.inverse_transform(cato_test_3[['Predicted_Target_premium_account']])
        
    mlflow.end_run()


    st.write(cato_test_3['Predicted_Target_premium_account'])




