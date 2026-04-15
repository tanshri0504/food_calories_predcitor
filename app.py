import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Safe import
try:
    from sklearn.linear_model import LinearRegression
except:
    st.error("⚠️ scikit-learn is not installed. Add it to requirements.txt")
    st.stop()

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Nutrition Analyzer Pro++", layout="wide")

# ---------------- THEME ----------------
theme = st.sidebar.toggle("🌙 Dark Mode")
if theme:
    st.markdown("<style>body{background-color:#0E1117;color:white;}</style>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🥗 AI Nutrition Analyzer Pro++")

# ---------------- DATA ----------------
data = {
    "Food": ["Rice","Chapati","Apple","Banana","Milk","Egg","Chicken","Paneer","Bread","Dal"],
    "Calories_per_unit": [130,120,95,105,42,78,239,265,80,150]
}
df = pd.DataFrame(data)

# ---------------- MODEL ----------------
df_model = pd.DataFrame({
    "Quantity": np.tile([1,2,3,4], len(df)),
    "Food": np.repeat(df["Food"], 4),
})
df_model["Calories"] = df_model.apply(
    lambda x: x["Quantity"] * df[df["Food"] == x["Food"]]["Calories_per_unit"].values[0], axis=1
)

df_encoded = pd.get_dummies(df_model, columns=["Food"])
X = df_encoded.drop("Calories", axis=1)
y = df_encoded["Calories"]

model = LinearRegression()
model.fit(X, y)

# ---------------- FUNCTIONS ----------------
def predict(food, qty):
    input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    input_data["Quantity"] = qty
    input_data[f"Food_{food}"] = 1
    return round(model.predict(input_data)[0], 2)

def bmi(weight, height):
    return round(weight / (height/100)**2, 2)

def bmi_status(b):
    if b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Normal"
    elif b < 30:
        return "Overweight"
    else:
        return "Obese"

def diet_suggestion(cal):
    if cal > 500:
        return "⚠️ Reduce fried foods, add salad"
    elif cal > 300:
        return "⚠️ Balanced but control portion"
    else:
        return "✅ Healthy diet"

def generate_pdf(name, cal, bmi_val):
    doc = SimpleDocTemplate("nutrition_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Calories: {cal}", styles["Normal"]))
    content.append(Paragraph(f"BMI: {bmi_val}", styles["Normal"]))

    doc.build(content)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ Food Input")
    name = st.text_input("Enter Name")
    food = st.selectbox("Select Food", df["Food"])
    qty = st.slider("Quantity", 1, 5, 1)

    st.subheader("⚖️ Body Details")
    weight = st.number_input("Weight (kg)", 30, 150, 60)
    height = st.number_input("Height (cm)", 100, 220, 165)

with col2:
    st.subheader("📊 Results")

    if st.button("Analyze Now"):
        calories = predict(food, qty)
        bmi_val = bmi(weight, height)
        status = bmi_status(bmi_val)

        # Save history
        st.session_state.history.append({
            "Time": datetime.datetime.now(),
            "Food": food,
            "Calories": calories
        })

        # -------- METRICS --------
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Calories", f"{calories} kcal")
        c2.metric("⚖️ BMI", bmi_val)
        c3.metric("📌 Status", status)

        # -------- SUGGESTION --------
        st.subheader("💡 Diet Suggestion")
        st.info(diet_suggestion(calories))

        # -------- CHART --------
        st.subheader("📈 Nutrition Chart")
        chart_df = pd.DataFrame({
            "Metric": ["Calories","BMI"],
            "Value": [calories, bmi_val]
        })
        st.bar_chart(chart_df.set_index("Metric"))

        # -------- PDF --------
        generate_pdf(name, calories, bmi_val)
        with open("nutrition_report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, "nutrition_report.pdf")

# ---------------- HISTORY ----------------
st.subheader("📜 History Tracker")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    st.line_chart(hist_df.set_index("Time")["Calories"])
else:
    st.info("No history yet")

# ---------------- ANALYTICS ----------------
st.subheader("📊 Food Analytics")

avg = hist_df["Calories"].mean() if st.session_state.history else 0
st.write(f"📊 Average Calories: {round(avg,2)}")

# ---------------- EXTRA ----------------
st.subheader("🚀 Pro Features")
st.success("✔ ML Prediction")
st.success("✔ BMI Calculator")
st.success("✔ Diet Suggestions")
st.success("✔ History Tracking")
st.success("✔ PDF Report")
