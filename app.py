import streamlit as st
import pandas as pd
import numpy as np
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Safe import
try:
    from sklearn.linear_model import LinearRegression
except:
    st.error("⚠️ scikit-learn is not installed. Add it to requirements.txt")
    st.stop()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Nutrition Analyzer Pro++", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: white;
}
.card {
    background-color: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🥗 AI Nutrition Analyzer Pro++")
st.caption("Smart Health Insights with AI 🚀")

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
        return "🔵 Underweight"
    elif b < 25:
        return "🟢 Normal"
    elif b < 30:
        return "🟠 Overweight"
    else:
        return "🔴 Obese"

def diet_suggestion(cal):
    if cal > 500:
        return "⚠️ Reduce fried foods, increase fiber"
    elif cal > 300:
        return "⚠️ Maintain balance & portion control"
    else:
        return "✅ Great! Keep it up"

def generate_pdf(name, cal, bmi_val):
    doc = SimpleDocTemplate("nutrition_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"<b>Name:</b> {name}", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"<b>Calories:</b> {cal}", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"<b>BMI:</b> {bmi_val}", styles["Normal"]))

    doc.build(content)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 🍽️ Enter Details")
    name = st.text_input("Name")
    food = st.selectbox("Food Item", df["Food"])
    qty = st.slider("Quantity", 1, 5, 1)

    st.markdown("### ⚖️ Body Metrics")
    weight = st.number_input("Weight (kg)", 30, 150, 60)
    height = st.number_input("Height (cm)", 100, 220, 165)

with col2:
    st.markdown("### 📊 Results Dashboard")

    if st.button("🚀 Analyze Now"):
        calories = predict(food, qty)
        bmi_val = bmi(weight, height)
        status = bmi_status(bmi_val)

        st.session_state.history.append({
            "Time": datetime.datetime.now(),
            "Food": food,
            "Calories": calories
        })

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Calories", f"{calories} kcal")
        m2.metric("⚖️ BMI", bmi_val)
        m3.metric("📌 Status", status)

        # Suggestion
        st.info(diet_suggestion(calories))

        # Chart
        chart_df = pd.DataFrame({
            "Metric": ["Calories","BMI"],
            "Value": [calories, bmi_val]
        })
        st.bar_chart(chart_df.set_index("Metric"))

        # PDF
        generate_pdf(name, calories, bmi_val)
        with open("nutrition_report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, "nutrition_report.pdf")

# ---------------- HISTORY ----------------
st.markdown("### 📜 History Tracker")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    st.line_chart(hist_df.set_index("Time")["Calories"])

    avg = hist_df["Calories"].mean()
    st.success(f"📊 Avg Calories: {round(avg,2)} kcal")
else:
    st.info("No data yet")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | AI Powered Nutrition Tracker")
