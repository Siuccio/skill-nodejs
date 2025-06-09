import os
import sqlite3
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict

# 1. Connect to the SQLite DB
base_dir = os.path.dirname(os.path.realpath(__file__))

db_path = os.path.join(base_dir, "..","..","..","mydb.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Elenca tutte le tabelle nel database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tabelle nel database:", tables)

# 2. Execute query to fetch flattened skill-value-level data
query = """
SELECT
  up.id AS user_project_id,
  up.level AS user_project_level,
  e.id AS evaluation_id,
  v.id AS value_id,
  v.value AS value_value,
  s.id AS skill_id,
  s.name AS skill_name
FROM user_project up
JOIN evaluation e ON e.userProjectId = up.id
JOIN value v ON v.evaluationId = e.id
JOIN skill s ON s.id = v.skillId
"""

df_raw = pd.read_sql(query, conn)

print(df_raw.head())

# Group by user_project_id and transform data
grouped = df_raw.groupby("user_project_id")
for project_id, group in grouped:
    print(f"User Project ID: {project_id}")
    print(group)

# 3. Transform to wide format: one row per user_project, columns = skills
skill_rows = []
grouped = df_raw.groupby("user_project_id")
for project_id, group in grouped:
    row = defaultdict(int)
    for _, row_data in group.iterrows():
        row[row_data["skill_name"].lower()] = row_data["value_value"]
        row["level"] = row_data["user_project_level"].lower()
    skill_rows.append(row)

df = pd.DataFrame(skill_rows).fillna(0)

# 4. Train model
X = df.drop(columns=["level"])
y = df["level"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save artifacts
model_dir = os.path.join(base_dir, "..", "models")
os.makedirs(model_dir, exist_ok=True)

dump(model, os.path.join(model_dir, "skill_level_model.joblib"))
dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

# Save feature names
features_path = os.path.join(model_dir, "features.json")
with open(features_path, "w") as f:
    json.dump(X.columns.tolist(), f)

# 6. Evaluation
accuracy = model.score(X_test, y_test)
print(f"✅ Modello addestrato con accuratezza: {accuracy:.2f}")
