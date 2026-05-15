
import pickle  # load saved model and scaler from .pkl files
from flask import Flask, request, jsonify, render_template 
 # Flask → app; request → get form data; jsonify → return JSON response; render_template → serve HTML files
import numpy as np       # numerical operations on input data
import pandas as pd        # data manipulation
from sklearn.preprocessing import StandardScaler # scale input data before prediction

application = Flask(__name__)
app = application

## importing ridge regressor and Standard scaler pickel
ridge_model = pickle.load(open('MODELS/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('MODELS/scaler.pkl', 'rb'))
# MODELS/ → subfolder where both .pkl files are stored


@app.route('/')
def index():
   return render_template('index.html') # serves the index.html page from the templates/ folder to the browser


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
   if request.method == "POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
   else:
      return render_template('home.html')

if __name__ == '__main__':       
   app.run(host = "0.0.0.0") 
# starts the Flask server
# host="0.0.0.0" → makes the app accessible on all network interfaces
# not just localhost — important for cloud deployment (AWS)