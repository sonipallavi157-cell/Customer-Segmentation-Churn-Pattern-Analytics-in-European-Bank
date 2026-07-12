import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Customer Churn Prediction")
st.write("Predict whether a bank customer will exit or not.")

# Load dataset
data = pd.read_csv("European_Bank.csv")

# Drop unnecessary columns
data = data.drop(["CustomerId", "Surname"], axis=1)

# Convert categorical columns
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data = pd.get_dummies(data, columns=["Geography"], drop_first=True)

# Features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.header("Enter Customer Details")

creditscore = st.number_input("Credit Score", 300, 900, 650)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
numproducts = st.number_input("Number of Products", 1, 4, 1)
hascard = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender = 1 if gender == "Male" else 0

input_data = pd.DataFrame(
    [[
        creditscore,
        gender,
        age,
        tenure,
        balance,
        numproducts,
        hascard,
        active,
        salary,
        geo_germany,
        geo_spain
    ]],
    columns=X.columns
)

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to Exit.")
    else:
        st.success("Customer is likely to Stay.")