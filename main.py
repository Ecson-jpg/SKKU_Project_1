import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("")

# Quick overview
print("Head:\n", df.head())
print("\nInfo:\n")
df.info()
print("\nDescribe:\n", df.describe())

# Clean up column names (The Kaggle dataset notoriously has trailing spaces like 'ALLERGY ')
df.columns = df.columns.str.strip()

# Set the visual style for seaborn
sns.set_theme(style="whitegrid")

# Create a figure with subplots to display all charts at once
plt.figure(figsize=(16, 12))

# 1. Box plot for Gender
plt.subplot(3, 2, 1)
sns.boxplot(data=df, x="GENDER", y="AGE", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Gender")

# 2. Scatter plot for Age
# Since Age is the only continuous variable, plotting it against the participant index
plt.subplot(3, 2, 2)
sns.scatterplot(data=df, x=df.index, y="AGE", hue="LUNG_CANCER", palette="Set2", alpha=0.7)
plt.title("Scatter Plot of Participant Ages")
plt.xlabel("Participant Index")

# 3. Box plot for Smoking habits
plt.subplot(3, 2, 3)
sns.boxplot(data=df, x="SMOKING", y="AGE", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Smoking Habit")
plt.xlabel("Smoking (1 = No, 2 = Yes)")

# 4. Box plot for Alcohol consumption
plt.subplot(3, 2, 4)
sns.boxplot(data=df, x="ALCOHOL CONSUMING", y="AGE", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Alcohol Consumption")
plt.xlabel("Alcohol Consuming (1 = No, 2 = Yes)")

# 5. Box plot for Allergy presence
plt.subplot(3, 2, 5)
sns.boxplot(data=df, x="ALLERGY", y="AGE", hue="LUNG_CANCER", palette="Set2")
plt.title("Age Distribution by Allergy Presence")
plt.xlabel("Allergy (1 = No, 2 = Yes)")

# Adjust layout so labels don't overlap and display the plots
plt.tight_layout()
plt.show()