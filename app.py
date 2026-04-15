import streamlit as st
import pandas as pd
import numpy as np

# Try importing sklearn safely
try:
    from sklearn.linear_model import LinearRegression
except:
    st.error("⚠️ scikit-learn is not installed. Please check requirements.txt")
    st.stop()

# Page config
st.set_page_config(page_title="Food Calorie Predictor", layout="wide")

# Title
st.title("🍔 Food Calorie Predictor")
st.write("Predict calories using Machine Learning")

# Dataset
data = {
    "Food": ["Rice","Rice","Chapati","Chapati","Apple","Apple","Banana","Banana","Milk","Milk","Egg","Egg","Chicken","Chicken"],
    "Quantity": [100,200,1,2,1,2,1,2,100,200,1,2,100,200],
    "Calories": [130,260,120,240,95,190,105,210,42,84,78,156,239,478]
}

df = pd.DataFrame(data)

# Encoding
df_encoded = pd.get_dummies(df, columns=["Food"])
X = df_encoded.drop("Calories", axis=1)
y = df_encoded["Calories"]

# Train model
model = LinearRegression()
model.fit(X, y)

# UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ Input")
    food = st.selectbox("Select Food", df["Food"].unique())
    quantity = st.slider("Quantity", 1, 300, 100)

with col2:
    st.subheader("📊 Result")

    if st.button("Predict"):
        # Prepare input
        input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
        input_data["Quantity"] = quantity
        input_data["Food_" + food] = 1

        # Prediction
        prediction = model.predict(input_data)[0]

        st.success(f"🔥 Calories: {round(prediction,2)} kcal")

        # Suggestion
        if prediction > 300:
            st.warning("⚠️ High calorie food")
        else:
            st.info("✅ Healthy choice")

        # Chart
        st.bar_chart(pd.DataFrame({"Calories": [prediction]}))

# Show dataset
st.subheader("📋 Dataset")
st.dataframe(df)
