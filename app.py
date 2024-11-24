from flask import Flask, request, jsonify
import numpy as np
import joblib


app = Flask(__name__)


model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Welcome to the Student Performance Predictor API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.json
        g1 = data.get("G1")
        g2 = data.get("G2")
        studytime = data.get("studytime")
        failures = data.get("failures")
        absences = data.get("absences")

        
        if None in [g1, g2, studytime, failures, absences]:
            return jsonify({"error": "Invalid input. Provide G1, G2, studytime, failures, and absences."}), 400

        
        trend = (g2 - g1) / g1 if g1 != 0 else 0

       
        input_data = np.array([g1, g2, studytime, failures, absences, trend]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        
        predicted_grade = model.predict(input_data_scaled)[0]
        return jsonify({
            "predicted_grade": round(predicted_grade, 2),
            "trend": "improving" if trend > 0 else "declining" if trend < 0 else "steady"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
