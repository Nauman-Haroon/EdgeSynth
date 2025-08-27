from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd

# Load SMOTE-balanced synthetic data
df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")
features = ['age', 'workclass', 'education', 'occupation', 'hours-per-week']
X = pd.get_dummies(df[features])
y = df['income']

# Encode labels from ['<=50K', '>50K'] → [0, 1]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # <=50K → 0, >50K → 1

# Save the encoder for later inference
dump(label_encoder, "models/xgb_label_encoder.joblib")

# Convert X to numpy array
X_np = X.to_numpy()

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_np, y_encoded)

# Save model and column order for ONNX export later
dump(model, "models/xgboost_model.joblib")
# After training
pd.DataFrame([X.columns.tolist()]).to_csv("data/xgboost_feature_order.csv", index=False, header=False)


print("✅ XGBoost model trained and saved successfully.")


# Load real test data
real_df = pd.read_csv("data/raw/adult.csv")
X_test = pd.get_dummies(real_df[features])
y_test = real_df["income"]

# Align columns
X_test, _ = X_test.align(X, join='right', axis=1, fill_value=0)
X_test_np = X_test.to_numpy()

# Encode real labels using saved encoder
y_test_encoded = label_encoder.transform(y_test)

# Predict
y_pred = model.predict(X_test_np)

# Evaluate
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\n📊 Accuracy on real test data: {accuracy}")
print("\n🧾 Classification Report:\n")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
