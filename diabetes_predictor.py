import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---- Dataset ----
data = {
    'Glucose': [85, 89, 120, 140, 150, 95, 130, 110, 100, 160],
    'BloodPressure': [66, 70, 80, 85, 90, 75, 88, 82, 78, 92],
    'BMI': [26.6, 28.1, 30.0, 32.5, 35.0, 27.0, 31.2, 29.5, 28.0, 36.5],
    'Age': [31, 29, 45, 50, 54, 33, 48, 40, 36, 60],
    'Outcome': [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# ---- Features & Target ----
X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Outcome']

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Model ----
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---- Prediction ----
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---- Custom Prediction ----
sample = pd.DataFrame([[130, 85, 31.0, 45]],
                      columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])

prediction = model.predict(sample)

print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")