import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Clean column names (IMPORTANT 🔥)
df.columns = df.columns.str.strip()

# Select features
features = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "WorkLifeBalance"
]

X = df[features]

# Convert target
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, "attrition_pipeline.pkl")

print("✅ Model saved successfully!")
