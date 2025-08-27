import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time


# Load ONNX model
session = ort.InferenceSession("models/xgboost_model.onnx")

# Prepare input
features = ['age', 'workclass', 'education', 'occupation', 'hours-per-week']
real_df = pd.read_csv("data/raw/adult.csv")

# One-hot encode features
X = pd.get_dummies(real_df[features])

# 🔧 Load saved feature order and reindex the test data accordingly
feature_order = pd.read_csv("data/xgboost_feature_order.csv", header=None).iloc[0].tolist()
X = X.reindex(columns=feature_order, fill_value=0)

# Convert to numpy array
X_np = X.astype(np.float32).to_numpy()

# Encode labels
y = real_df["income"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Run inference
input_name = session.get_inputs()[0].name
preds = session.run(None, {input_name: X_np})[0]

start_time = time.time()

preds = session.run(None, {input_name: X_np})[0]

end_time = time.time()
print(f"⚡ ONNX Inference Time: {(end_time - start_time):.4f} seconds")


# Evaluate
print("🧾 ONNX Inference Classification Report:\n")
print(classification_report(y_encoded, preds, target_names=le.classes_))
