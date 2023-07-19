import joblib
import pandas as pd
import psycopg2
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 1. get data
db_connect = psycopg2.connect(
    host="localhost",
    database="mydatabase",
    user="myuser",
    password="mypassword"
)
df = pd.read_sql(
    "SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect
)
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2023)

# 2. model development and train
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_val)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_val, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
joblib.dump(model_pipeline, "db_pipeline.joblib")

# 4. save data
df.to_csv("data.csv", index=False)


