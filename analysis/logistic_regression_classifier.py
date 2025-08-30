# train_synth_test_real.py
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTENC    



# Load datasets
real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/synthetic_adult.csv")

print(synthetic_df["income"].value_counts())


# Use same feature columns
features = ['age', 'workclass', 'education', 'occupation', 'hours-per-week']  # Adjust if needed
target = 'income'

X = synthetic_df[features]
y = synthetic_df[target]

# Categorical feature indices
categorical_features = [1, 2, 3]  # positions in the features list

# Apply SMOTENC
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)


# One-hot encode after resampling
X_resampled_encoded = pd.get_dummies(X_resampled)
y_resampled_encoded = y_resampled

# Prepare real data
X_test = pd.get_dummies(real_df[features])
y_test = real_df[target]

# Align columns between train and test
X_resampled_encoded, X_test = X_resampled_encoded.align(X_test, join='left', axis=1, fill_value=0)

# Train and evaluate
model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
model.fit(X_resampled_encoded, y_resampled_encoded)
y_pred = model.predict(X_test)

print("Train on synthetic (SMOTENC) → Test on real")
print(classification_report(y_test, y_pred))

balanced_synthetic = X_resampled.copy()
balanced_synthetic['income'] = y_resampled
balanced_synthetic.to_csv("data/synthetic/smote_synthetic_adult.csv", index=False)


# After training
dump(model, "models/logistic_model.joblib")
