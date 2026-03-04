import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('Data/corona_tested_individuals_ver_006.english.csv')

# Display basic information about the DataFrame
print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())
