import os

logistic_path = "models/logistic_model.joblib"
logistic_onnx_path = "models/logistic_model.onnx"
xgboost_model_path = "models/xgboost_model.joblib"
xgboost_onnx_path = "models/xgboost_model.onnx"

print("🔍 File Sizes (in KB):")
for path in [logistic_path, logistic_onnx_path, xgboost_model_path, xgboost_onnx_path]:
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"{path}: {size_kb:.2f} KB")
    else:
        print(f"{path} not found.")
