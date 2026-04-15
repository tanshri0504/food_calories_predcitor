import streamlit as st
import pandas as pd
import numpy as np

# Safe import
try:
    from sklearn.linear_model import LinearRegression
except:
    st.error("⚠️ scikit-learn is not installed. Please add it in requirements.txt")
    st.stop()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Food Calorie Predictor Pro+", layout="wide")

# ---------------- THEME ----------------
theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    st.markdown(
        "<style>body{background-color:#0E1117;color:white;}</style>",
        unsafe_allow_html=True
    )

# ---------------- TITLE ----------------
st.title("🍔 AI Food Calorie Predictor Pro+")

# ---------------- DATASET ----------------
data = {
    "Food": ["Rice","Rice","Chapati","Chapati","Apple","Apple",
             "Banana","Banana","Milk","Milk","Egg","Egg",
             "Chicken","Chicken"],
    "Quantity": [100,200,1,2,1,2,1,2,100,200,1,2,100,200],
    "Calories": [130,260,120,240,95,190,105,210,42,84,78,156,239,478]
}

df = pd.DataFrame(data)

# ---------------- MODEL TRAINING ----------------
df_encoded = pd.get_dummies(df, columns=["Food"])
X = df_encoded.drop("Calories", axis=1)
y = df_encoded["Calories"]

model = LinearRegression()
model.fit(X, y)

# ---------------- FUNCTIONS ----------------
def predict_calories(food, quantity):
    input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    input_data["Quantity"] = quantity

    if "Food_" + food in input_data.columns:
        input_data["Food_" + food] = 1

    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

def health_suggestion(calories):
    if calories > 400:
        return "⚠️ Very High Calories", "warning"
    elif calories > 250:
        return "⚠️ Moderate Calories", "warning"
    else:
        return "✅ Healthy Choice", "success"

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ Food Input")
    food = st.selectbox("Select Food", df["Food"].unique())
    quantity = st.slider("Quantity", 1, 300, 100)

with col2:
    st.subheader("📊 Prediction Result")

    if st.button("Predict Calories"):
        calories = predict_calories(food, quantity)
        suggestion, status = health_suggestion(calories)

        # -------- METRICS --------
        c1, c2 = st.columns(2)
        c1.metric("🔥 Calories", f"{calories} kcal")
        c2.metric("🍽️ Quantity", quantity)

        # -------- RESULT --------
        if status == "warning":
            st.warning(suggestion)
        else:
            st.success(suggestion)

        # -------- CHART --------
        st.subheader("📊 Calorie Visualization")
        chart_df = pd.DataFrame({
            "Type": ["Calories"],
            "Value": [calories]
        })
        st.bar_chart(chart_df.set_index("Type"))

# ---------------- DATASET VIEW ----------------
st.subheader("📋 Dataset Preview")
st.dataframe(df)

# ---------------- INSIGHTS ----------------
st.subheader("📌 Smart Insights")

avg_cal = df["Calories"].mean()
max_cal = df["Calories"].max()

st.write(f"📊 Average Calories: **{round(avg_cal,2)} kcal**")
st.write(f"🔥 Highest Calories Food: **{df.loc[df['Calories'].idxmax(),'Food']}** ({max_cal} kcal)")

# ---------------- TIPS ----------------
st.subheader("💡 Health Tips")

st.info("🥗 Prefer fruits and low-calorie foods daily")
st.info("🏃 Exercise regularly to balance calorie intake")
st.info("💧 Drink enough water")
