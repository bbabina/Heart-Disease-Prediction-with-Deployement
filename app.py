from click import style
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def predict():

    age=int(request.form['age'])
    sex=(request.form['sex'])
    chest_pain_type= (request.form['chest_pain_type'])
    resting_bp=int(request.form['resting_bp'])
    cholesterol=int(request.form['cholesterol'])
    fasting_bs=int(request.form['fasting_bs'])
    resting_ecg=(request.form['resting_ecg'])
    max_hr=int(request.form['max_hr'])
    exercise_angina=(request.form['exercise_angina'])
    old_peak=float(request.form['old_peak'])
    st_slope=(request.form['st_slope'])
    
    chestpain_map = {'1': 'TA', '2': 'ASY', '3': 'ATA', '4': 'NAP'}
    resting_ecg_map = {'1': 'Normal', '2': 'ST', '3': 'LVH'}
    st_slope_map = {'1': 'Up', '2': 'Flat', '3': 'Dowm'}
    
    data = {'Age': [age],
            'Sex': ['M' if sex == '1' else 'F' ],
            'ChestPainType': [chestpain_map[chest_pain_type]],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg_map[resting_ecg]],
            'MaxHR': [max_hr],
            'ExerciseAngina': ['Y' if exercise_angina == '1' else 'N'],
            'Oldpeak': [old_peak],
            'ST_Slope': [st_slope_map[st_slope]],
            }
    df = pd.DataFrame(data)
    df['Zero_Cholesterol'] = df['Cholesterol'] == 0
    df['Zero_RestingBP'] = df['RestingBP'] == 0

    df['Cholesterol']=df['Cholesterol'].replace(0,np.nan)
    df['RestingBP']=df['RestingBP'].replace(0,np.nan)

    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")

    threshold = 0.2
    Y_pred = (model.predict_proba(pipeline.transform(df))[0])[1] > threshold

    return render_template("predict.html", my_prediction = Y_pred)


if __name__ == "__main__":
    app.run(debug=True)
