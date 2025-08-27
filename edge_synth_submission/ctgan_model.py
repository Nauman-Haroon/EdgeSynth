import pandas as pd
from ctgan import CTGAN
from sklearn.utils import resample
import torch
import pickle  


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Drop rows with missing values for simplicity
    df = df.dropna()

    # Clean string data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Remove rows with missing values
    df = df[~df.isin(['?', ' ?']).any(axis=1)]


    # Optional: Reduce to relevant columns
    df = df[['age', 'workclass', 'education', 'occupation', 'hours-per-week', 'income']]

    
    low_income = df[df['income'] == '<=50K']
    high_income = df[df['income'] == '>50K']

    if len(high_income) > 0 and len(low_income) > 0:
        high_income_up = resample(high_income, replace=True, n_samples=len(low_income), random_state=42)
        df = pd.concat([low_income, high_income_up])

    

    print("Processed data shape:", df.shape)
    print("Income class distribution:\n", df['income'].value_counts())

    return df



def train_ctgan(data, discrete_columns):
    print("Starting training...")
    print(f"Training CTGAN on data shape: {data.shape}, with discrete columns: {discrete_columns}")

    if not discrete_columns:
        print("⚠️ No discrete (categorical) columns specified. Proceeding with numeric-only training.")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ctgan = CTGAN(
    embedding_dim=128,                  
    generator_dim=(256, 256),           
    discriminator_dim=(256, 256),
    batch_size=128,
    epochs=1000,                        
    pac=2,                              
    cuda=torch.cuda.is_available(),
    verbose=True
    )

    
    ctgan.fit(data, discrete_columns=discrete_columns)
    print("Training complete.")
    return ctgan

def generate_synthetic_data(model, n_samples=1000):
    return model.sample(n_samples)
    
