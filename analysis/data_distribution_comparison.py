import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


real_df = pd.read_csv("data/raw/adult.csv")
synthetic_df = pd.read_csv("data/synthetic/smote_synthetic_adult.csv")


cols_to_plot = ['age', 'hours-per-week', 'education', 'workclass']

# we will create subplots (2 rows x 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]

    # Categorical columns which are side-by-side bars
    if real_df[col].dtype == 'object':
        # Get value counts
        real_counts = real_df[col].value_counts()
        synthetic_counts = synthetic_df[col].value_counts()
        
        # we filter out the missing categories
        common_categories = list(set(real_counts.index) & set(synthetic_counts.index))
        
        # Filter the counts to only include these common categories
        real_vals = [real_counts[cat] for cat in common_categories]
        synthetic_vals = [synthetic_counts[cat] for cat in common_categories]

        x = range(len(common_categories))
        width = 0.35
        ax.bar([p for p in x], real_vals, width=width, label="Real", alpha=0.7, color="blue")
        ax.bar([p + width for p in x], synthetic_vals, width=width, label="Synthetic", alpha=0.7, color="red")
        ax.set_xticks([p + width / 2 for p in x])
        ax.set_xticklabels(common_categories, rotation=45, ha="right")
        ax.set_ylabel("Count")

    # histograms
    else:
        sns.histplot(real_df[col], label="Real", color="blue", kde=True, stat="density", ax=ax)
        sns.histplot(synthetic_df[col], label="Synthetic", color="red", kde=True, stat="density", ax=ax)

    ax.set_title(f"Distribution of {col}")
    ax.legend()

plt.tight_layout()
plt.show()
