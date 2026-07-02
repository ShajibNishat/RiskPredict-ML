import joblib
import numpy as np

# Load model
model = joblib.load('risk_model.pkl')
risk_map_reverse = {0: 'Low', 1: 'Medium', 2: 'High'}

print("\n🔮 IT Project Risk Predictor (CLI)")
print("Enter project details below:\n")

duration = float(input("Duration (days): "))
team_size = float(input("Team size (persons): "))
budget = float(input("Budget (K USD): "))
past_delays = float(input("Past delay frequency (0-1): "))
req_changes = float(input("Requirement changes (count): "))

features = np.array([[duration, team_size, budget, past_delays, req_changes]])
pred_code = model.predict(features)[0]
risk_level = risk_map_reverse[pred_code]
confidence = np.max(model.predict_proba(features)[0])

print(f"\n📊 Predicted Risk Level: {risk_level}")
print(f"   Confidence: {confidence:.2f}")