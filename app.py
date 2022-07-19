from click import style
from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def predict():

    age=float(request.form['age'])
    sex=float(request.form['sex'])
    chest_pain_type= float(request.form['chest_pain_type'])
    resting_bp=float(request.form['resting_bp'])
    cholesterol=float(request.form['cholesterol'])
    fasting_bs=float(request.form['fasting_bs'])
    resting_ecg=float(request.form['resting_ecg'])
    max_hr=float(request.form['max_hr'])
    exercise_angina=float(request.form['exercise_angina'])
    old_peak=float(request.form['old_peak'])
    st_slope=float(request.form['st_slope'])
    

    
    X= np.array([[ age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, old_peak, st_slope ]])
    
    model_path=r'C:\Users\Epoch\OneDrive\Desktop\heart-disease-deployement\models\rfTuning1.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X)
    

    return render_template("predict.html", my_prediction = Y_pred)


if __name__ == "__main__":
    app.run(debug=True)
