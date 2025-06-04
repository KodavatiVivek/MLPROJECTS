import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
#SKLEARN
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformation_config:
    preprocessor_obj_file_path : str=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformation_config()

    def get_data_transformer_object(self):
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipleine = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False, with_std=False))  # StandardScaler without centering
            ])

            logging.info("Numerical and categorical pipelines created successfully")
            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipleine, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor
        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data= pd.read_csv(train_path)
            test_data= pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            logging.info("Preprocessing data...")
            preposser_obj= self.get_data_transformer_object()

            target_column_name = 'math_score'

            input_feature_train_df = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info("Fitting and transforming training data and test data done...")

            input_feature_train_arr= preposser_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preposser_obj.transform(input_feature_test_df)
            logging.info("Data transformation completed successfully")   
            # Using numpy to concatenate(np.c_) the input features and target feature
            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Train and test arrays created successfully")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preposser_obj
            )

            logging.info("Preprocessor object saved successfully")
            logging.info(f"Preprocessor object file path: {self.config.preprocessor_obj_file_path}")

            # Saving the preprocessor object
            return (train_arr,test_arr,self.config.preprocessor_obj_file_path)

            
        except Exception as e:
            logging.error(f"Error reading train or test data: {str(e)}")
            raise CustomException(e, sys)