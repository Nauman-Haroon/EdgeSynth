import os
import pandas as pd
from ctgan_model import load_and_preprocess_data, train_ctgan, generate_synthetic_data
import pickle

real_data_path = "data/raw/adult.csv"
output_path = "data/synthetic/synthetic_adult.csv"

# Load and preprocess
df = load_and_preprocess_data(real_data_path)

# Define categorical (discrete) columns
discrete_columns = ['workclass', 'education', 'occupation', 'income']
# discrete_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Train CTGAN
ctgan_model = train_ctgan(df, discrete_columns)

#Save model
with open("models/ctgan_model_2.pkl", "wb") as f:
    pickle.dump(ctgan_model, f)

# Generate synthetic data
synthetic_df = generate_synthetic_data(ctgan_model, 5000)

# Save the output
os.makedirs("data/synthetic", exist_ok=True)
synthetic_df.to_csv(output_path, index=False)

print("Synthetic data saved to:", output_path)
