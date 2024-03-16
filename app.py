from flask import Flask, render_template,request
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    sl = float(request.form.get("slcm"))
    sw = float(request.form.get("swcm"))
    pl = float(request.form.get("plcm"))
    pw = float(request.form.get("pwcm"))
    data_point = np.array([[sl,sw,pl,pw]])
    model = pickle.load(open(r"D:\iris\iris.pkl",'rb'))
    prediction = model.predict(data_point)
    return render_template("home.html",prediction=prediction)










if __name__=="__main__":
    app.run(debug=True)

