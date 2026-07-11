import pandas as pd
from scipy.stats import ttest_ind

# data load
df = pd.read_csv('survey lung cancer.csv')
df.columns = df.columns.str.strip()

# Convert cancer occurrence status into numerical values (1, 0).
df['LUNG_CANCER_VAL'] = df['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)

# Setting Analysis Variables
symptoms = ['ANXIETY', 'FATIGUE', 'WHEEZING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
habits = ['SMOKING', 'YELLOW_FINGERS', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'ALCOHOL CONSUMING']


# t-test function for generating execution and result tables
def get_ttest_result(group1, group2, variables, col1_name, col2_name):
    results = []
    for var in variables:
        g1_data = group1[var]
        g2_data = group2[var]
        stat, p_val = ttest_ind(g1_data, g2_data, nan_policy='omit')

        results.append({
            'Variable': var,
            col1_name: round(g1_data.mean(), 2),
            col2_name: round(g2_data.mean(), 2),
            'p-value': round(p_val, 4),
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })
    return pd.DataFrame(results)


# --- 1. Differences in overall variables by gender ---
males = df[df['GENDER'] == 'M']
females = df[df['GENDER'] == 'F']
all_vars = ['AGE'] + symptoms + habits
print("\n[Analysis 1] Differences in overall variables by gender")
print(get_ttest_result(males, females, all_vars, 'Male Avg', 'Female Avg').to_string(index=False))

# --- 2. Comparison of symptoms by gender among cancer patients (YES) ---
cancer_df = df[df['LUNG_CANCER'] == 'YES']
males_yes = cancer_df[cancer_df['GENDER'] == 'M']
females_yes = cancer_df[cancer_df['GENDER'] == 'F']
print("\n\n[Analysis 2] Symptom differences by gender among cancer patients (YES)")
print(get_ttest_result(males_yes, females_yes, symptoms, 'Male Avg', 'Female Avg').to_string(index=False))

# --- 3. Differences in Cancer Incidence Based on Lifestyle Habits ---
results_habits = []
for var in habits:
    group_no = df[df[var] == 1]['LUNG_CANCER_VAL']
    group_yes = df[df[var] == 2]['LUNG_CANCER_VAL']
    stat, p_val = ttest_ind(group_no, group_yes, nan_policy='omit')

    results_habits.append({
        'Habit': var,
        'No(1) Cancer Rate': round(group_no.mean(), 2),
        'Yes(2) Cancer Rate': round(group_yes.mean(), 2),
        'p-value': round(p_val, 4),
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
print("\n\n[Analysis 3] Differences in Cancer Incidence Based on Lifestyle Habits")
print(pd.DataFrame(results_habits).to_string(index=False))