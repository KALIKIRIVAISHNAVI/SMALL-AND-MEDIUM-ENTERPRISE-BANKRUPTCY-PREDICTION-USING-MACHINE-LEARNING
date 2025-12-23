import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load Model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_bankruptcy():
    try:
        # 1. Get data
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        # 2. Scale
        final_features_scaled = scaler.transform(final_features)
        
        # 3. Predict Probability
        probability = model.predict_proba(final_features_scaled)[0][1]
        risk_score = round(probability * 100, 2)
        
        # --- EXPERT SYSTEM LAYER (The Safety Net) ---
        # If the financial ratios are disastrous, we FORCE a high risk score.
        # This overrides the ML model if it gets confused.
        
        # Rule 1: Negative Working Capital (Feature 0) is a massive red flag.
        if features[0] < -0.05: 
            risk_score = max(risk_score, 85.0) # Force at least 85% risk
            
        # Rule 2: Insolvent (Debt > Assets) (Feature 4 is Debt/TA)
        if features[4] > 1.0:
            risk_score = max(risk_score, 90.0) # Force at least 90% risk
            
        # Rule 3: Deeply Unprofitable (EBIT/TA < -0.1) (Feature 2)
        if features[2] < -0.1:
            risk_score = max(risk_score, 80.0)

        # 4. Generate Result
        # We lower the threshold to 30% because in finance, 30% risk is ALREADY dangerous.
        if risk_score > 30:
            prediction_text = "High Risk of Bankruptcy"
            alert_color = "danger"
            explanation = (f"CRITICAL ALERT: The calculated probability of distress is {risk_score}%. "
                           "Key indicators such as Liquidity or Debt-to-Assets ratios are at dangerous levels.")
        else:
            prediction_text = "Low Risk (Healthy)"
            alert_color = "success"
            explanation = (f"Healthy Status: The risk probability is low ({risk_score}%). "
                           "The company shows stable financial health based on current inputs.")

        return render_template('index.html', 
                               prediction_text=prediction_text, 
                               explanation=explanation, 
                               risk_score=risk_score, 
                               alert_color=alert_color)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", alert_color="warning")

if __name__ == "__main__":
    app.run(debug=True)