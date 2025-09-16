# src/train.py

import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Ensure the dataset is placed in `data/Resume.csv`
df = pd.read_csv("data/Resume.csv")

# Dataset usually has 'Category' (target) and 'Resume' (text)
print("Dataset Shape:", df.shape)
print("Categories:", df['Category'].unique())

# -------------------------------
# 2. Predefined Skills List
# -------------------------------
# This dictionary is populated based on Gaurav Dutta Kaggle dataset
# For each role, we define a set of relevant skills
# Updated role_skills with top 5 skills per role
role_skills = {
    "Data Science": ["python", "pandas", "numpy", "scikit-learn", "sql"],
    "HR": ["recruitment", "talent acquisition", "interviewing", "onboarding", "hr policies"],
    "Arts Teacher": ["creativity", "communication", "drawing", "painting", "art history"],
    "Web Designer": ["html", "css", "javascript", "ux/ui design", "responsive design"],
    "Mechanical Engineer": ["solidworks", "autocad", "matlab", "mechanics", "thermodynamics"],
    "Sales": ["negotiation", "crm", "communication", "lead generation", "sales strategy"],
    "Health and Fitness Trainer": ["exercise planning", "nutrition", "communication", "client management", "fitness assessment"],
    "Civil Engineer": ["autocad", "structural analysis", "surveying", "project management", "construction materials"],
    "Java Developer": ["java", "spring", "hibernate", "sql", "oop"],
    "Python Developer": ["python", "django", "flask", "api development", "oop"],
    "Full Stack Developer": ["javascript", "react", "nodejs", "sql", "api development"],
    "Frontend Developer": ["html", "css", "javascript", "react", "responsive design"],
    "Database Engineer": ["sql", "mysql", "postgresql", "database design", "performance tuning"],
    "DevOps Engineer": ["docker", "kubernetes", "ci/cd", "aws", "linux"],
    "Network Security Engineer": ["firewall", "penetration testing", "network protocols", "linux", "vpn"],
    "Ethical Hacker": ["penetration testing", "network security", "python", "linux", "exploitation"],
    "Business Analyst": ["excel", "sql", "power bi", "tableau", "data analysis"],
    "Automation Tester": ["selenium", "test cases", "automation", "java", "python"],
    "PMO": ["project management", "planning", "communication", "risk management", "reporting"],
    "Blockchain Developer": ["solidity", "ethereum", "smart contracts", "nodejs", "cryptography"],
    "ETL Developer": ["sql", "python", "data warehousing", "etl tools", "oracle"],
    "SAP Developer": ["sap abap", "sap modules", "integration", "sql", "business processes"]
}

# Save role_skills for app.py
with open("src/role_skills.pkl", "wb") as f:
    pickle.dump(role_skills, f)

print("✅ Role-Skills dictionary saved successfully in src/role_skills.pkl")

# -------------------------------
# 3. Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)                # remove URLs
    text = re.sub(r"[^a-zA-Z ]", " ", text)           # remove numbers/special chars
    text = text.lower()                                # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)                  # remove extra spaces
    return text.strip()

df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X = df["Cleaned_Resume"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000, stop_words="english", ngram_range=(1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 6. Model Training
# -------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# -------------------------------
# 7. Evaluation
# -------------------------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Save Model & Vectorizer
# -------------------------------
with open("src/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("src/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and Vectorizer saved successfully in src/")
