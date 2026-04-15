import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Food Calorie Predictor", page_icon="🍔", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: grey;
    }
    .box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<p class="title">🍔 Food Calorie Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict calories using Machine Learning</p>', unsafe_allow_html=True)

st.write("")

# ------------------ DATASET ------------------
data = {
    "Food": ["Rice","Rice","Chapati","Chapati","Apple","Apple","Banana","Banana","Milk","Milk","Egg","Egg","Chicken","Chicken"],
    "Quantity": [100,200,1,2,1,2,1,2,100,200,1,2,100,200],
    "Calories": [130,260,120,240,95,190,105,210,42,84,78,156,239,478]
}

df = pd.DataFrame(data)

# ------------------ MODEL TRAINING ------------------
df_encoded = pd.get_dummies(df, columns=["Food"])
X = df_encoded.drop("Calories", axis=1)
y = df_encoded["Calories"]

model = LinearRegression()
model.fit(X, y)

# ------------------ LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("🍽️ Enter Food Details")

    food = st.selectbox("Select Food", df["Food"].unique())
    quantity = st.slider("Select Quantity", 1, 300, 100)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if st.button("🔍 Predict Calories"):
        # Prepare input
        input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
        input_data["Quantity"] = quantity
        input_data["Food_" + food] = 1

        prediction = model.predict(input_data)[0]

        st.success(f"🔥 Estimated Calories: {round(prediction,2)} kcal")

        # Health suggestion
        if prediction > 300:
            st.warning("⚠️ High calorie food! Eat in moderation.")
        else:
            st.info("✅ Healthy choice!")

        # Chart
        chart_data = pd.DataFrame({
            "Calories": [prediction]
        })
        st.bar_chart(chart_data)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ EXTRA DASHBOARD ------------------
st.write("")
st.subheader("📈 Dataset Overview")

st.dataframe(df)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("👨‍💻 Developed using Streamlit + Machine Learning")
