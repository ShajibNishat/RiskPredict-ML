import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Generate features
duration = np.random.randint(10, 180, n_samples)          # days
team_size = np.random.randint(1, 15, n_samples)           # persons
budget = np.random.randint(50, 1000, n_samples)           # K USD
past_delay_freq = np.random.uniform(0, 1, n_samples)      # 0 = never delayed, 1 = always
req_changes = np.random.randint(0, 50, n_samples)         # number of requirement changes

# Generate risk label (High/Medium/Low) based on rules + noise
# High risk: long duration, small team, low budget, high past delays, many changes
score = (duration / 180) * 0.3 \
        + (1 - team_size/15) * 0.2 \
        + (1 - budget/1000) * 0.2 \
        + past_delay_freq * 0.2 \
        + (req_changes/50) * 0.1

noise = np.random.normal(0, 0.05, n_samples)
score = np.clip(score + noise, 0, 1)

def assign_risk(s):
    if s < 0.4:
        return "Low"
    elif s < 0.7:
        return "Medium"
    else:
        return "High"

risk_label = [assign_risk(s) for s in score]

# Create DataFrame
df = pd.DataFrame({
    'duration_days': duration,
    'team_size': team_size,
    'budget_k_usd': budget,
    'past_delay_frequency': past_delay_freq,
    'requirement_changes': req_changes,
    'risk_level': risk_label
})

df.to_csv('project_data.csv', index=False)
print("✅ Synthetic dataset saved as 'project_data.csv'")
print(f"   Shape: {df.shape}")
print(df.head())