import pandas as pd
import numpy as np
from joblib import load
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import time

# Load model 
start = time.time()
model = load("models/xgboost_model.joblib")
end = time.time()
print(f"✅ Model loaded in {round(end - start, 2)} seconds")


# Load feature order used during training
feature_order = pd.read_csv("data/xgboost_feature_order.csv", header=None).iloc[0].tolist()

# Use real data to extract dummy input
features = ['age', 'workclass', 'education', 'occupation', 'hours-per-week']
X_sample = pd.read_csv("data/raw/adult.csv")[features]
X_sample_encoded = pd.get_dummies(X_sample)
X_sample_encoded, _ = X_sample_encoded.align(pd.DataFrame(columns=feature_order), join='right', axis=1, fill_value=0)

# Convert to numpy and float32 (required)
X_np = X_sample_encoded.to_numpy().astype(np.float32)
initial_type = [('float_input', FloatTensorType([None, X_np.shape[1]]))]

# === Convert to ONNX ===
onnx_model = convert_xgboost(model.get_booster(), initial_types=initial_type)
onnx.save_model(onnx_model, "models/xgboost_model.onnx")
print("✅ XGBoost model exported to ONNX format → models/xgboost_model.onnx")
