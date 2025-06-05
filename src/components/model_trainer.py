from dataclasses import dataclass
import os
import sys

import pandas as pd
import numpy as np

# SKLEARN ALGORITHMS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

# SKLEARN METRICS
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def train_model(self, train_arr, test_arr,prepossor_path):
        try:
           #Load Train and Test Data
           logging.info("Loading train and test data")
           X_train,y_train = train_arr[:,:-1], train_arr[:,-1]
           X_test,y_test = test_arr[:,:-1], test_arr[:,-1]

           #Modelas
           models = {
               'LinearRegression': LinearRegression(),
               'RandomForestRegressor': RandomForestRegressor(),
               'AdaBoostRegressor': AdaBoostRegressor(),
               'GradientBoostingRegressor': GradientBoostingRegressor(),
               'DecisionTreeRegressor': DecisionTreeRegressor(),
               'SVR': SVR(),
               'KNeighborsRegressor': KNeighborsRegressor(),
               'CatBoostRegressor': CatBoostRegressor(verbose=0)
           }

           params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
           logging.info("Models initialized successfully")

           model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,models= models,params=params)
           logging.info("Models evaluated successfully with hyperparameter tuning")
           logging.info(f"Model report: {model_report}")

           logging.info("Evaluating models to find the best one")

           # TO get Best Model Score on each model
           #next(iterator, default_value)
           best_model_score = max(X['r2_score'] for X in model_report.values())
           logging.info(f"Best model r2 score: {best_model_score}")
           best_model_name = next(
           name for name, metrics in model_report.items() if metrics['r2_score'] == best_model_score
         )
           logging.info("Best model found")       
           if best_model_score < 0.6:
                raise CustomException("No best model found with r2 score greater than 0.6", sys)
           logging.info(f"Best model name: {best_model_name}")

           save_object(file_path=self.config.model_path, obj=models[best_model_name])
           logging.info(f"Best model {best_model_name} saved successfully at {self.config.model_path}")

           predictions = models[best_model_name].predict(X_test)
           r2 = r2_score(y_test, predictions)
           return r2
        except Exception as e:
            logging.error(f"Error in loading preprocessor object: {str(e)}")
            raise CustomException(e, sys)



