import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

first_names = [
    "Amina","David","Grace","John","Fatima","Samuel","Esther","Daniel",
    "Joy","Michael","Ruth","Emmanuel","Mary","Joshua","Sarah","Victor",
    "Deborah","Paul","Mercy","Isaac","Blessing","Peter","Naomi","Philip",
    "Hannah","James","Rebecca","Stephen","Andrew","Juliet","Timothy","Dorcas",
    "Moses","Priscilla","Caleb","Benjamin","Elizabeth","Nathan","Sophia","Joseph",
    "Daniela","Matthew","Faith","Samuel","Linda","Kelvin","Oluwaseun","Chidinma",
    "Mariam","Abdul"
]

last_names = [
    "Adeyemi","Okafor","Bello","Ibrahim","Ojo","Okoro","Uche","Nwachukwu",
    "Adebayo","Olawale","Abdullahi","Garba","Balogun","Chukwu","Nnaji",
    "Sani","Yakubu","Mohammed","Eze","Obi","Onyeka","Ajayi","Lawal",
    "Ogunleye","Ogunbiyi","Ogunlana","Akinyemi","Adesina","Oseni","Afolabi",
    "Adewale","Akanbi","Olatunji","Oshodi","Nwosu","Umeh","Ndukwe","Musa",
    "Aliyu","Yusuf","Ahmed","Danjuma","Usman","Kalu","Anyanwu","Ibeh",
    "Alabi","Ibekwe","Osei","Peters"
]

departments = [
    "Computer Science",
    "AI Engineering",
    "Data Science",
    "Software Engineering"
]

rows = []

for i in range(1, 101):

    attendance = random.randint(60, 100)

    study_hours = round(random.uniform(3, 16), 1)

    assignment = max(40, min(100, int(study_hours*5 + random.randint(15, 30))))
    quiz = max(40, min(100, assignment + random.randint(-8, 8)))
    exam = max(40, min(100, assignment + random.randint(-10, 10)))

    final = round((assignment*0.2)+(quiz*0.2)+(exam*0.6),1)

    rows.append({
        "Student_ID": f"STU{i:03}",
        "Name": random.choice(first_names) + " " + random.choice(last_names),
        "Gender": random.choice(["Male","Female"]),
        "Age": random.randint(18,24),
        "Department": random.choice(departments),
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Assignment_Score": assignment,
        "Quiz_Score": quiz,
        "Exam_Score": exam,
        "Final_Score": final,
        "Pass": "Yes" if final >= 50 else "No"
    })

df = pd.DataFrame(rows)

# ---------- Missing Values ----------
for col in ["Attendance","Study_Hours","Quiz_Score"]:
    idx = np.random.choice(df.index, 3, replace=False)
    df.loc[idx, col] = np.nan

# ---------- Duplicate Rows ----------
duplicates = df.sample(2, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# ---------- Outliers ----------
df.loc[5, "Study_Hours"] = 30
df.loc[18, "Attendance"] = 25

df.to_csv("student_performance.csv", index=False)

print("✅ student_performance.csv created successfully!")
print(df.head())