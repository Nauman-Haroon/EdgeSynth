import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import load
import time


# Load data
real_df = pd.read_csv("data/raw/adult.csv")
features = ['age', 'workclass', 'education', 'occupation', 'hours-per-week']
target = 'income'

# Prepare test set
real_df = real_df[features + [target]]
X_test = pd.get_dummies(real_df[features])
y_test = real_df[target]

# Align features for consistency
model_features = pd.get_dummies(pd.read_csv("data/synthetic/smote_synthetic_adult.csv")[features])
X_test = X_test.reindex(columns=model_features.columns, fill_value=0)

# Load trained model
model = load("models/logistic_model.joblib")

# Convert model to ONNX
initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open("models/logistic_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Logistic model exported to ONNX")

# Run inference using ONNXRuntime
session = ort.InferenceSession("models/logistic_model.onnx")
input_name = session.get_inputs()[0].name


# Convert test data to float32
X_test_np = X_test.astype(np.float32).to_numpy()
pred_onx = session.run(None, {input_name: X_test_np})[0]

start_time = time.time()
pred_onx = session.run(None, {input_name: X_test_np})[0]
end_time = time.time()
print(f"⚡ ONNX Inference Time: {(end_time - start_time):.4f} seconds")

# Evaluate performance
print("\n🔍 ONNX Inference Classification Report:")
print(classification_report(y_test, pred_onx))

# Evaluate performance
print("\n🔍 ONNX Inference Classification Report:")
print(classification_report(y_test, pred_onx))
