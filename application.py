from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

#utils
from src.pipeline.Predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler

application= Flask(__name__)

app= application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template ('predict.html')
        # Get the input values from the form
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        logging.info("Data received from the form")
        pred_df=data.get_data_as_df()
        logging.info(f"DataFrame created: {pred_df}")
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict_data(pred_df)
        logging.info(f"Prediction results: {results}")

        return render_template('predict.html', results=results[0])


if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080)        
    app.run(host='0.0.0.0', port=8080)
    print("Starting the application...")  # Debugging line to confirm app start
    logging.info("Application started successfully")  # Log the start of the application
    print(f"link: http://localhost:8080")  # Debugging line to confirm the link
    logging.info("Application is running on http://localhost:8080")  # Log the link
