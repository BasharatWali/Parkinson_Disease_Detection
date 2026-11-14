# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:18:33 2023

@author: abash
"""

from flask import Flask, render_template, request
import numpy as np
import nibabel as nib
import pickle
#from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from radiomics import featureextractor
import pandas as pd
import matplotlib.pyplot as plt
import os
from tempfile import NamedTemporaryFile


app = Flask(__name__)

mask = "PD25-subcortical-1mm_bmask.nii.gz"
# Load the trained model
# Load the classification model
with open('models/classification_model.pkl', 'rb') as model_file:
    classification = pickle.load(model_file)

# Load the feature_selection model
with open('models/feature_selection_model.pkl', 'rb') as model:
    feature_selection = pickle.load(model)

# Load the MinMaxScaler and LabelEncoder
with open('models/scalar.pkl', 'rb') as model:
    scalar = pickle.load(model)
#le = LabelEncoder()

# Initialize PyRadiomics feature extractor
params = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'enableCExtensions': True,
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        # Create a temporary file and save the uploaded file data
        base, extension1 = os.path.splitext(file.filename)
        _, extension2 =  os.path.splitext(base)
        file_extension = extension2 + extension1
        temp_file = NamedTemporaryFile(suffix=file_extension, delete=False)
        file.save(temp_file.name)
        

        radiomics_result = extractor.execute(temp_file.name, mask)
        filtered_dict = {k: v for k, v in radiomics_result.items() if isinstance(v, (np.float64, np.ndarray)) and (v.dtype == 'float64' if isinstance(v, np.ndarray) else True)}


        df = pd.DataFrame(filtered_dict, index=[0])

        df = df.iloc[:, 17:]

        features_array = np.array(df)
        features_scaled = scalar.transform(features_array)
        features_transformed = feature_selection.transform(features_scaled)

        prediction = classification.predict(features_transformed)
        classes = ['Healthy', 'Parkinson']
        img = nib.load(temp_file.name)
        # Get the middle slice
        middle_slice = img.get_fdata()[:, :, img.shape[2] // 2]
        if not os.path.exists('static'):
            os.makedirs('static')
        # Save the middle slice as a PNG image
        plt.imsave('static/middle_slice.png', middle_slice, cmap='gray')
        temp_file.close()
        # Delete the temporary file
        os.unlink(temp_file.name)
        
        return render_template('index.html', prediction_text=f"Predicted Class is: {classes[int(prediction)]}")
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
