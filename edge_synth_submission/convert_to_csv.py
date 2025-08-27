import pandas as pd

# Define column names from the dataset documentation
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Read the raw .data file
df = pd.read_csv('data/raw/adult.data', header=None, names=columns)

# Clean whitespace
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Save as CSV
df.to_csv('data/raw/adult.csv', index=False)

print("Converted adult.data to adult.csv successfully!")
