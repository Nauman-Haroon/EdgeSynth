import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


synthetic_df = pd.read_csv("data/synthetic/synthetic_adult.csv")        
balanced_df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")   


fig, axes = plt.subplots(1, 2, figsize=(12, 5))


sns.countplot(ax=axes[0], x='income', data=synthetic_df, palette='Blues')
axes[0].set_title("Class Distribution Before SMOTENC")
axes[0].set_xlabel("Income Class")
axes[0].set_ylabel("Count")


sns.countplot(ax=axes[1], x='income', data=balanced_df, palette='Greens')
axes[1].set_title("Class Distribution After SMOTENC")
axes[1].set_xlabel("Income Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
