import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Customer Churn Prediction")
st.write("Predict whether a bank customer will exit or stay.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("European_Bank.csv")
    return data

data = load_data()

# Data preprocessing
data = data.drop(["Year", "CustomerID", "Surname"], axis=1, errors="ignore")

# Convert Gender into numerical values
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# Convert Geography column
data = pd.get_dummies(data, columns=["Geography"], drop_first=True)

# Features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

st.sidebar.write("Model Accuracy:")
st.sidebar.write(round(accuracy * 100, 2), "%")


# User Input Section
st.header("Enter Customer Details")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    max_value=300000.0,
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_credit_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    max_value=300000.0,
    value=50000.0
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)


# Convert user input into model format

gender_value = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0


input_data = pd.DataFrame(
    [[
        credit_score,
        gender_value,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        active_member,
        salary,
        geo_germany,
        geo_spain
    ]],
    columns=X.columns
)


# Prediction button

if st.button("Predict"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Exit.")
        st.write(
            "Churn Probability:",
            round(probability * 100, 2),
            "%"
        )

    else:
        st.success("✅ Customer is likely to Stay.")
        st.write(
            "Churn Probability:",
            round(probability * 100, 2),
            "%"
        )
