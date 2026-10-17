from flask import Flask, request, jsonify
from pipeline.trainer import Trainer
from core.logistic_regression import LogisticRegressionModel

app = Flask(__name__)
model = LogisticRegressionModel()
trainer = Trainer(model)
trainer.train()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model.predict([data])
    return jsonify({"prediction": float(prediction[0])})