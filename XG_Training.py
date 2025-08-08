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
import pickle
from sklearn.model_selection import GridSearchCV



dx = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/train.csv')
dz = pd.read_csv('C:/Users/Hp/Downloads/playground-series-s4e12/test.csv')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("XGR_Boost_Regressor_1")

with mlflow.start_run(run_name="MinmaxX"):
       
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
        for cato_train_1_1 in list(train_out)[0:2]:
           categorical_columns_cato_train_2.append(cato_train_1_1)
            
        cato_train_38 = mima(cato_train_1,categorical_columns_cato_train_2)
        cato_train_3 = cato_train_38.min_max()

        A = cato_train_3.drop('Premium Amount', axis=1)
        b = np.log1p(cato_train_3["Premium Amount"])
        A = A[sorted(A.columns)]

        

    
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2 , random_state = 42)

        param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
         }
        
        XGR_Boost = XGBRegressor()
        grid_search = GridSearchCV(XGR_Boost, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(A_train,b_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        

        

        y_pred_log = best_model.predict(A_test)
        B_pred = np.expm1(y_pred_log)
        signature = infer_signature(A_train, best_model.predict(A_train))
        
        mae = mean_absolute_error(np.expm1(b_test),B_pred)
        mse = mean_squared_error(np.expm1(b_test), B_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(np.expm1(b_test), B_pred)
        name = "Minmax"
        model = XGR_Boost

        mlflow.log_params(best_params)
        mlflow.log_param("Scaler", name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2 Score", r2)
        mlflow.sklearn.log_model(
        sk_model=XGR_Boost,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="XGR_Boost",
    )
        
filename = 'C:/Users/Hp/OneDrive/Desktop/smart premium model/XGR_model_1.pkl'
pickle.dump(XGR_Boost, open(filename, 'wb'))
        
        
mlflow.end_run()