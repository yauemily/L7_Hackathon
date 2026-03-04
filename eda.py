import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('Data/corona_tested_individuals_ver_006.english.csv')

print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Basic information
print("\n1. Dataset Overview")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Missing values
print("\n2. Missing Values")
print(df.isnull().sum())

# Data types
print("\n3. Data Types")
print(df.dtypes)

# Statistical summary
print("\n4. Statistical Summary")
print(df.describe())

# Corona result distribution
print("\n5. Corona Result Distribution")
print(df['corona_result'].value_counts())
print(f"\nPositive rate: {(df['corona_result'] == 'positive').sum() / len(df) * 100:.2f}%")

# Gender distribution
print("\n6. Gender Distribution")
print(df['gender'].value_counts())

# Age distribution
print("\n7. Age 60+ Distribution")
print(df['age_60_and_above'].value_counts())

# Test indication
print("\n8. Test Indication Distribution")
print(df['test_indication'].value_counts())

# Symptoms analysis
symptoms = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']
print("\n9. Symptoms Prevalence")
for symptom in symptoms:
    count = df[symptom].sum()
    pct = count / len(df) * 100
    print(f"{symptom}: {count} ({pct:.2f}%)")

# Visualizations
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Corona Dataset EDA', fontsize=16)

# 1. Corona result distribution
df['corona_result'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Corona Test Results')
axes[0, 0].set_ylabel('Count')

# 2. Gender distribution
df['gender'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'pink'])
axes[0, 1].set_title('Gender Distribution')
axes[0, 1].set_ylabel('Count')

# 3. Age distribution
df['age_60_and_above'].value_counts().plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Age 60+ Distribution')
axes[0, 2].set_ylabel('Count')

# 4. Test indication
df['test_indication'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Test Indication')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Symptoms comparison
symptom_counts = [df[symptom].sum() for symptom in symptoms]
axes[1, 1].bar(symptoms, symptom_counts)
axes[1, 1].set_title('Symptoms Prevalence')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Corona result by gender
pd.crosstab(df['gender'], df['corona_result']).plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Corona Result by Gender')
axes[1, 2].set_ylabel('Count')
axes[1, 2].legend(title='Result')

# 7. Corona result by age group
pd.crosstab(df['age_60_and_above'], df['corona_result']).plot(kind='bar', ax=axes[2, 0])
axes[2, 0].set_title('Corona Result by Age Group')
axes[2, 0].set_ylabel('Count')
axes[2, 0].legend(title='Result')

# 8. Symptoms in positive cases
positive_df = df[df['corona_result'] == 'positive']
positive_symptoms = [positive_df[symptom].sum() for symptom in symptoms]
axes[2, 1].bar(symptoms, positive_symptoms, color='red', alpha=0.7)
axes[2, 1].set_title('Symptoms in Positive Cases')
axes[2, 1].set_ylabel('Count')
axes[2, 1].tick_params(axis='x', rotation=45)

# 9. Correlation heatmap
numeric_cols = symptoms + ['corona_result']
corr_df = df.copy()
corr_df['corona_result'] = (corr_df['corona_result'] == 'positive').astype(int)
correlation = corr_df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 2])
axes[2, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n10. Visualizations saved as 'eda_visualizations.png'")
plt.show()

print("\n" + "="*50)
print("EDA COMPLETE")
print("="*50)
