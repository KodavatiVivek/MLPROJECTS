import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import pickle

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