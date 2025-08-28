import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load real and synthetic datasets
real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")

# Get numeric columns that exist in BOTH datasets
numeric_cols = list(set(real_df.select_dtypes(include=[np.number]).columns)
                    & set(synthetic_df.select_dtypes(include=[np.number]).columns))

# Sort the columns for consistent comparison
numeric_cols.sort()

# Compute correlation matrices only for the common numeric columns
real_corr = real_df[numeric_cols].corr()
synthetic_corr = synthetic_df[numeric_cols].corr()

# Create side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap for real data
sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=False, cbar=True)
axes[0].set_title("Real Data Correlation Heatmap")

# Heatmap for synthetic data
sns.heatmap(synthetic_corr, ax=axes[1], cmap="coolwarm", annot=False, cbar=True)
axes[1].set_title("Synthetic Data Correlation Heatmap")

plt.tight_layout()
plt.show()
