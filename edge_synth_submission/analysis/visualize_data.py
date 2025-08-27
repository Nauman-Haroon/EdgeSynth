import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")

cols_to_plot = ['age', 'hours-per-week', 'education', 'workclass']

for col in cols_to_plot:
    plt.figure(figsize=(10, 4))

    if real_df[col].dtype == 'object':
        sns.countplot(y=real_df[col], color='blue', alpha=0.5, label='Real')
        sns.countplot(y=synthetic_df[col], color='red', alpha=0.5, label='Synthetic')
    else:
        sns.histplot(real_df[col], color='blue', label='Real', kde=True, stat='density')
        sns.histplot(synthetic_df[col], color='red', label='Synthetic', kde=True, stat='density')

    plt.title(f"Comparison of {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()
