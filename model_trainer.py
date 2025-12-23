import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import pickle

# 1. Load Data
try:
    df = pd.read_csv('sme_bankruptcy_10k_v2.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: csv file not found. Make sure 'sme_bankruptcy_10k_v2.csv' is in the folder.")
    exit()

# 2. Prepare Features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. SMOTE (Fix Imbalance)
# We force it to create even MORE bankrupt examples to learn from
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Gradient Boosting (The "Smarter" Model)
print("Training Gradient Boosting Model...")
model = GradientBoostingClassifier(
    n_estimators=200,     # More trees
    learning_rate=0.1,    # Learn faster
    max_depth=5,          # Deeper logic
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
print("\nModel Evaluation:")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 8. Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("SUCCESS: New 'model.pkl' and 'scaler.pkl' created.")