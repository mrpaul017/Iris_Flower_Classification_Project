import numpy as np
import pickle as pkl
from flask import Flask, request, jsonify, render_template

#creating a flask app
flask_app=Flask(__name__)
model=pkl.load(open('final model.pkl', 'rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))
if __name__=="__main__":
    flask_app.run(debug=True)
    