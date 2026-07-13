import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ---------------------------------------------------------
# 1. DATA LOADING & ENGINEERING
# ---------------------------------------------------------
# Load the NLST dataset
df = pd.read_csv("nlst_780_prsn_idc_20210527.csv")

# Clean up column names just in case
df.columns = df.columns.str.strip()

# Engineer the Target Variable:
# If 'candx_days' is not null, the patient was diagnosed with cancer.
df['LUNG_CANCER'] = df['candx_days'].notna().map({True: 'YES', False: 'NO'})
df['LUNG_CANCER_VAL'] = df['candx_days'].notna().astype(int)

# Quick overview
print("Head:\n", df[['pid', 'age', 'gender', 'cigsmok', 'candx_days', 'LUNG_CANCER']].head())
print("\nTarget Variable Distribution:\n", df['LUNG_CANCER'].value_counts())

# ---------------------------------------------------------
# 2. VISUALIZATIONS
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 10))

# 1. Box plot for Gender vs Age
plt.subplot(2, 2, 1)
# Note: In NLST, gender is typically numeric (1=Male, 2=Female)
sns.boxplot(data=df, x="gender", y="age", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Gender")

# 2. Scatter plot for Age vs Participant Index
plt.subplot(2, 2, 2)
# Using 'pid' (Participant ID) instead of the dataframe index for a cleaner x-axis
sns.scatterplot(data=df, x="pid", y="age", hue="LUNG_CANCER", palette="Set2", alpha=0.4)
plt.title("Scatter Plot of Participant Ages")
plt.xlabel("Participant ID (pid)")

# 3. Box plot for Smoking habits
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x="cigsmok", y="age", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Baseline Smoking (cigsmok)")
plt.xlabel("Smoking Status (e.g., 1 = Yes, 0 = No)")

# 4. Box plot for Race
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x="race", y="age", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Race")
plt.xlabel("Race Category")

# Adjust layout and save
plt.tight_layout()
plt.savefig("nlst_clinical_visuals.pdf", format="pdf", bbox_inches="tight")
print("\n✅ Visualizations successfully saved to 'nlst_clinical_visuals.pdf'")

# ---------------------------------------------------------
# 3. STATISTICAL ANALYSIS (T-TESTS)
# ---------------------------------------------------------
# We will test continuous/numeric clinical variables that exist in the NLST 'prsn' file
clinical_vars = ['age', 'race', 'cigsmok', 'canc_free_days']

def get_ttest_result(group1, group2, variables, col1_name, col2_name):
    results = []
    for var in variables:
        # Convert to numeric, forcing errors to NaN, then drop NaNs for clean testing
        g1_data = pd.to_numeric(group1[var], errors='coerce').dropna()
        g2_data = pd.to_numeric(group2[var], errors='coerce').dropna()
        
        # Skip if a group has no data for this variable
        if len(g1_data) == 0 or len(g2_data) == 0:
            continue

        stat, p_val = ttest_ind(g1_data, g2_data, nan_policy='omit')

        results.append({
            'Variable': var,
            col1_name: round(g1_data.mean(), 2),
            col2_name: round(g2_data.mean(), 2),
            'p-value': round(p_val, 4),
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })
    return pd.DataFrame(results)

# --- Analysis 1: Differences in overall variables by gender ---
males = df[df['gender'] == 1]
females = df[df['gender'] == 2]

print("\n[Analysis 1] Differences in overall clinical variables by gender (1 vs 2)")
print(get_ttest_result(males, females, clinical_vars, 'Gender 1 Avg', 'Gender 2 Avg').to_string(index=False))

# --- Analysis 2: Comparison by gender among cancer patients (YES) ---
cancer_df = df[df['LUNG_CANCER'] == 'YES']
males_yes = cancer_df[cancer_df['gender'] == 1]
females_yes = cancer_df[cancer_df['gender'] == 2]

print("\n\n[Analysis 2] Differences by gender among confirmed cancer patients")
print(get_ttest_result(males_yes, females_yes, clinical_vars, 'Gender 1 Avg', 'Gender 2 Avg').to_string(index=False))

# --- Analysis 3: Differences in Cancer Incidence Based on Baseline Smoking ---
# Comparing cancer rates between baseline smokers (1) and non-smokers (0)
group_no_smoke = df[df['cigsmok'] == 0]['LUNG_CANCER_VAL']
group_smoke = df[df['cigsmok'] == 1]['LUNG_CANCER_VAL']

stat, p_val = ttest_ind(group_no_smoke, group_smoke, nan_policy='omit')

results_smoking = [{
    'Habit': 'Baseline Smoking (1 vs 0)',
    'No (0) Cancer Rate': round(group_no_smoke.mean(), 4),
    'Yes (1) Cancer Rate': round(group_smoke.mean(), 4),
    'p-value': round(p_val, 4),
    'Significant': 'Yes' if p_val < 0.05 else 'No'
}]

print("\n\n[Analysis 3] Differences in Cancer Incidence Based on Baseline Smoking Status")
print(pd.DataFrame(results_smoking).to_string(index=False))