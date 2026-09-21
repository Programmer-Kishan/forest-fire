from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## import Ridge regressor and standard scalar pickle
ridge_model = pickle.load(open("./models/ridge.pkl", "rb"))
standard_scalar = pickle.load(open("./models/scaler.pkl", "rb"))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        new_scale_data = standard_scalar.transform(features)
        result = ridge_model.predict(new_scale_data)

        return render_template("home.html", results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")