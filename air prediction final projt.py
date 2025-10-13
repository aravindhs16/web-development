# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

# Load the Random Forest Classifier model
filename = 'randomforest.pkl'
classifier = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model file: {e}")
    print("Using fallback prediction method...")
    classifier = None

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        t = int(request.form['T'])
        tm = int(request.form['TM'])
        tmm = int(request.form['Tm'])
        slp = int(request.form['SLP'])
        h = int(request.form['H'])
        vv = float(request.form['VV'])
        v = float(request.form['V'])
        vm = int(request.form['VM'])
        
        if classifier is not None:
            # Use the trained model
            data = np.array([[t, tm, tmm, slp, h, vv, v, vm]])
            my_prediction = classifier.predict(data)
        else:
            # Fallback prediction based on simple heuristics
            # This is a basic estimation - not as accurate as the trained model
            base_aqi = 50  # Base AQI
            
            # Temperature effect (higher temps can increase pollution)
            temp_factor = (t - 20) * 0.5
            
            # Humidity effect (high humidity can trap pollutants)
            humidity_factor = (h - 50) * 0.3
            
            # Wind effect (higher wind can disperse pollutants)
            wind_factor = -(v - 10) * 0.4
            
            # Visibility effect (lower visibility often indicates higher pollution)
            visibility_factor = -(vv - 10) * 2
            
            # Pressure effect
            pressure_factor = (slp - 1013) * 0.01
            
            estimated_aqi = base_aqi + temp_factor + humidity_factor + wind_factor + visibility_factor + pressure_factor
            
            # Ensure AQI is within reasonable bounds
            estimated_aqi = max(0, min(500, estimated_aqi))
            my_prediction = np.array([estimated_aqi])
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)