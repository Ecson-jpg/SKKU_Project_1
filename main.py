import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv("survey lung cancer.csv")

print(df.head(),
      df.info(),
      df.describe(),
    )

columns = ["GENDER","AGE","SMOKING","ANXIETY","CHRONIC DISEASE","FATIGUE", "ALLERGY", "WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]
result = ["LUNG_CANCER"]


