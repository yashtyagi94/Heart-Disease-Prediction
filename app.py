from flask import Flask, request, render_template
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load model
model = xgb.Booster()
model.load_model("xgboost_model.json")


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        data = {
            "age": float(request.form["age"]),
            "sex": float(request.form["sex"]),
            "chest pain type": float(request.form["cp"]),
            "resting bp s": float(request.form["trestbps"]),
            "cholesterol": float(request.form["chol"]),
            "fasting blood sugar": float(request.form["fbs"]),
            "resting ecg": float(request.form["restecg"]),
            "max heart rate": float(request.form["thalach"]),
            "exercise angina": float(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "ST slope": float(request.form["slope"])
        }

        # Feature order must match training time
        feature_order = [
            "age", "sex", "chest pain type", "resting bp s", "cholesterol",
            "fasting blood sugar", "resting ecg", "max heart rate",
            "exercise angina", "oldpeak", "ST slope"
        ]

        df = pd.DataFrame([data], columns=feature_order)
        dmatrix = xgb.DMatrix(df)

        # Prediction
        pred_prob = model.predict(dmatrix)[0]
        result = int(pred_prob > 0.5)

        # Confidence level
        if pred_prob >= 0.8:
            danger_level = "High"
        elif pred_prob >= 0.6:
            danger_level = "Medium"
        else:
            danger_level = "Low"

        # Return styled result
        if result:
            return f"""<h2 style='color:red; text-align:center;'>Prediction: Heart Disease Detected</h2>
                       <h3 style='color:red; text-align:center;'>Confidence: {danger_level}</h3>"""
        else:
            return f"""<h2 style='color:lightgreen; text-align:center;'>Prediction: No Heart Disease</h2>"""


    except Exception as e:
        return f"<h3>Error: {e}</h3>"


if __name__ == '__main__':
    app.run(debug=True)
