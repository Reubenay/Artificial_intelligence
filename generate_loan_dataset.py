"""
generate_loan_dataset.py

Generates a realistic, intentionally messy ~1000-row "loan_approval.csv"
dataset for Week 6 - Day 1 (Perceptron & ANN).

Run this file once BEFORE opening the notebook:

    python generate_loan_dataset.py

It will create "loan_approval.csv" in the same folder. The dataset is
deliberately messy (missing values, inconsistent text casing, duplicate
rows, and a few outliers) so that students practice a real-world
cleaning workflow before training a Neural Network, the same way a
real AI project would.
"""

import numpy as np
import pandas as pd

# A fixed random seed makes the "randomness" repeatable, so every
# student who runs this script gets the exact same dataset.
np.random.seed(42)

N = 1000  # number of loan applicants to simulate

# ---------------------------------------------------------------
# 1. Simulate the raw features of each loan applicant
# ---------------------------------------------------------------

genders = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])

married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])

dependents = np.random.choice([0, 1, 2, 3], size=N, p=[0.45, 0.25, 0.20, 0.10])

education = np.random.choice(
    ["Graduate", "Not Graduate"], size=N, p=[0.7, 0.3]
)

self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.15, 0.85])

# Monthly income in Naira (skewed distribution - most people earn
# moderate amounts, a few earn a lot).
applicant_income = np.random.lognormal(mean=10.8, sigma=0.55, size=N).round(2)

# Some applicants also have a co-applicant income (0 if none).
coapplicant_income = np.where(
    np.random.rand(N) < 0.45,
    np.random.lognormal(mean=9.8, sigma=0.6, size=N).round(2),
    0.0,
)

# Loan amount requested (in thousands of Naira).
loan_amount = (
    (applicant_income + coapplicant_income) * np.random.uniform(0.08, 0.25, size=N)
).round(2)

loan_term_months = np.random.choice(
    [120, 180, 240, 300, 360], size=N, p=[0.1, 0.15, 0.2, 0.25, 0.3]
)

credit_score = np.random.normal(loc=650, scale=80, size=N).round(0)
credit_score = np.clip(credit_score, 300, 850)

property_area = np.random.choice(
    ["Urban", "Semiurban", "Rural"], size=N, p=[0.4, 0.35, 0.25]
)

years_employed = np.random.exponential(scale=5, size=N).round(1)
years_employed = np.clip(years_employed, 0, 35)

existing_loans = np.random.choice([0, 1, 2], size=N, p=[0.6, 0.3, 0.1])

# ---------------------------------------------------------------
# 2. Build a target "Loan_Approved" that follows realistic patterns
#    (not pure randomness) so the Neural Network has real signal to learn.
# ---------------------------------------------------------------

# A monthly-payment-to-income ratio (a standard real-world affordability
# signal banks actually use) instead of raw, unscaled amounts.
monthly_payment = loan_amount / loan_term_months
total_monthly_income = (applicant_income + coapplicant_income) + 1.0  # avoid /0
payment_to_income_ratio = monthly_payment / total_monthly_income

raw_score = (
    0.006 * (credit_score - 650)
    - 3000.0 * payment_to_income_ratio
    - 0.45 * existing_loans
    + 0.06 * years_employed
    + np.where(education == "Graduate", 0.35, -0.25)
    + np.where(married == "Yes", 0.15, 0.0)
)

# Centre the score around 0 so roughly half the applicants sit above a
# reasonable approval threshold, then add a little noise so the boundary
# is not perfectly clean (real approvals are never perfectly separable).
score = raw_score - np.median(raw_score)
probability = 1 / (1 + np.exp(-score))
loan_approved = (probability > np.random.uniform(0.35, 0.55, size=N)).astype(int)

# ---------------------------------------------------------------
# 3. Assemble the clean DataFrame first
# ---------------------------------------------------------------

df = pd.DataFrame(
    {
        "Applicant_ID": [f"LN{100000 + i}" for i in range(N)],
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term_months,
        "Credit_Score": credit_score,
        "Years_Employed": years_employed,
        "Existing_Loans": existing_loans,
        "Property_Area": property_area,
        "Loan_Approved": loan_approved,
    }
)

# ---------------------------------------------------------------
# 4. Now deliberately "dirty" the dataset, the way real-world data
#    collected from forms/branches usually looks.
# ---------------------------------------------------------------

# 4a. Missing values scattered across several columns
for col, frac in [
    ("Gender", 0.03),
    ("Self_Employed", 0.05),
    ("Credit_Score", 0.04),
    ("Loan_Amount", 0.02),
    ("Years_Employed", 0.03),
    ("Dependents", 0.02),
]:
    idx = df.sample(frac=frac, random_state=hash(col) % (2**32)).index
    df.loc[idx, col] = np.nan

# 4b. Inconsistent text casing / spacing (common in real form data)
messy_gender_idx = df.sample(frac=0.05, random_state=1).index
df.loc[messy_gender_idx, "Gender"] = df.loc[messy_gender_idx, "Gender"].str.lower()

messy_married_idx = df.sample(frac=0.05, random_state=2).index
df.loc[messy_married_idx, "Married"] = df.loc[messy_married_idx, "Married"].apply(
    lambda x: f" {x} " if pd.notna(x) else x
)

messy_area_idx = df.sample(frac=0.05, random_state=3).index
df.loc[messy_area_idx, "Property_Area"] = df.loc[
    messy_area_idx, "Property_Area"
].str.upper()

# 4c. A handful of unrealistic outliers (data-entry errors)
outlier_idx = df.sample(n=6, random_state=4).index
df.loc[outlier_idx[:3], "Applicant_Income"] = df["Applicant_Income"].max() * np.random.uniform(
    5, 9, size=3
)
df.loc[outlier_idx[3:], "Loan_Amount"] = df["Loan_Amount"].max() * np.random.uniform(6, 10, size=3)

# 4d. Duplicate rows (e.g. an applicant's form submitted twice)
duplicate_rows = df.sample(n=20, random_state=5)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 4e. Shuffle so duplicates and dirty rows aren't clustered together
df = df.sample(frac=1.0, random_state=6).reset_index(drop=True)

# ---------------------------------------------------------------
# 5. Save to CSV
# ---------------------------------------------------------------

df.to_csv("loan_approval.csv", index=False)

print(f"Saved loan_approval.csv with {len(df)} rows and {df.shape[1]} columns.")
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nLoan_Approved value counts:")
print(df["Loan_Approved"].value_counts())
