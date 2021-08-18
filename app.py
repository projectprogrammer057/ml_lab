import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = breast_cancer_detector_model.predict(df)
        
    if output == 0:
        res_val = "malignant tumor,The Patient may be affected in Breast Cancer.Please Consult to the oncologist as soon as possible"
    else:
        res_val = "benign tumor,The Patient have no Breast Cancer.Please Consult to the Doctor for taking treatment of benign tumor"
        

    return render_template('index.html', prediction_text='The Tumor is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
