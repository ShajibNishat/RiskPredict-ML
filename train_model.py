import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('project_data.csv')

# Encode target
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_code'] = df['risk_level'].map(risk_map)

# Features and target
features = ['duration_days', 'team_size', 'budget_k_usd', 'past_delay_frequency', 'requirement_changes']
X = df[features]
y = df['risk_code']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Save model
joblib.dump(model, 'risk_model.pkl')
print("✅ Model saved as 'risk_model.pkl'")