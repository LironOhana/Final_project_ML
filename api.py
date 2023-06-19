import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
rf_model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
scaler_y = joblib.load('scaler_y.pkl')

@app.route('/')
def home():
    return render_template('property_form.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')

    final_features = np.array(features, dtype=float).reshape(1, -1)
    final_features_scaled = scaler.transform(final_features)
    prediction = rf_model.predict(final_features_scaled)[0]
    print(prediction)
    prediction = np.array(prediction).reshape(1, -1)
    prediction= scaler_y.inverse_transform(prediction)[0][0]
    #risk = rf_model.predict_proba(final_features)[0][1]
    output_text = prediction

    return render_template('property_form.html', prediction_text='{}'.format(output_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)