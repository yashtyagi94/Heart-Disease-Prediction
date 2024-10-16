from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model (make sure to have the model in the same directory or provide the correct path)
model = joblib.load("/kaggle/working/")

@app.route('/')
def home():
    return render_template("chat.py/designe.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = [float(x) for x in request.form.values()]
    input_features = np.array([form_data])
    
    # Predict using the model
    prediction = model.predict(input_features)
    
    # Render the prediction on the same page
    return render_template("chat.py/designe.html", prediction_text=f"Heart Disease Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

if __name__ == "__main__":
    app.run(debug=True)
