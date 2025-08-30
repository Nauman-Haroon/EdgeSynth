import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")

# I can use 'age' or 'hours-per-week'

col = 'age' 

plt.figure(figsize=(10, 6))
sns.kdeplot(real_df[col], label='Real Data', fill=True)
sns.kdeplot(synthetic_df[col], label='Synthetic Data', fill=True)
plt.title(f'Distribution Comparison for {col}')
plt.legend()
plt.show()
