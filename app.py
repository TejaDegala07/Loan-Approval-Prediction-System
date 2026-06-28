from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = [
        float(request.form["Gender"]),
        float(request.form["Married"]),
        float(request.form["Dependents"]),
        float(request.form["Education"]),
        float(request.form["Self_Employed"]),
        float(request.form["ApplicantIncome"]),
        float(request.form["CoapplicantIncome"]),
        float(request.form["LoanAmount"]),
        float(request.form["Loan_Amount_Term"]),
        float(request.form["Credit_History"]),
        float(request.form["Property_Area"])
    ]

    prediction = model.predict([data])

    result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)