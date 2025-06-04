import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple regression models and return a report of their performance.
    """
    try:
        model_report = {}
        for i in range(len(list(models))):
            model_name= list(models.keys())[i]
            model = models[model_name]
            #Fir the model on train data
            model.fit(X_train, y_train)
            #Predict on train data
            y_train_pred = model.predict(X_train)
            #Predict on test data
            y_test_pred = model.predict(X_test)
            #Calculate metrics
            r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            #Store metrics in report
            model_report[model_name] = {
                'r2_score': r2,
                'mean_absolute_error': mae,
                'mean_squared_error': mse
            }

            return model_report
        # Sort the model report based on r2_score
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise CustomException(e, sys)