import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor n standard scaler pickle
ridge_model = pickle.load(open('flask_app/models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('flask_app/models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('home.html')



@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float (request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float (request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')












if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)