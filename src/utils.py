import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException(e, sys)
    
    

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

def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    """
    Evaluate multiple regression models and return a report of their performance.
    """
    try:
        model_report = {}
        for i in range(len(list(models))):
            model_name= list(models.keys())[i]
            logging.info(f"Evaluating model: {model_name}")
            #Get the model
            if model_name in params:
                para = params[model_name]
                model = models[model_name]
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=2)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                logging.info(f"No specific parameters for {model_name}, using default settings.")
                model = models[model_name]
                model.fit(X_train, y_train)
            #Log the best parameters found by GridSearchCV
            

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