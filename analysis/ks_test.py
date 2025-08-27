from scipy.stats import ks_2samp
import pandas as pd

real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/synthetic_adult.csv")

num_cols = ['age', 'hours-per-week']

for col in num_cols:
    stat, p_value = ks_2samp(real_df[col], synthetic_df[col])
    print(f"{col} → KS Statistic = {stat:.4f}, p-value = {p_value:.4f}")
