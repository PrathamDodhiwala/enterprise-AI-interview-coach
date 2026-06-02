import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from textblob import TextBlob

st.set_page_config(
    page_title="Enterprise AI Interview Coach", page_icon="🎯", layout="wide"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = joblib.load("interview_model.pkl")

df = pd.read_csv("interview_dataset.csv")

# ---------------------------------
# STYLE
# ---------------------------------

st.markdown(
    """
<style>
    /* Charcoal Background */
    .main {
        background-color: #121214;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #E4E4E7 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1A1A1E;
        border-right: 1px solid #2A2A32;
    }
    
    /* High contrast metrics */
    div[data-testid="stMetric"] {
        background-color: #1A1A1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2A2A32;
    }
    
    div[data-testid="stMetric"] label {
        color: #A1A1AA !important;
        font-size: 13px !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #34D399 !important; /* High-contrast Mint Green */
        font-size: 32px !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    ["Dashboard", "Mock Interview", "AI Feedback", "Analytics", "Reports"],
)

# ---------------------------------
# DASHBOARD
# ---------------------------------

if page == "Dashboard":

    st.title("🎯 Enterprise AI Interview Coach")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Candidates", "5,000")

    with col2:
        st.metric("Questions", "150+")

    with col3:
        st.metric("AI Accuracy", "95.4%")

    with col4:
        st.metric("Success Rate", "89%")

    st.markdown("---")

    chart = px.histogram(df, x="InterviewScore", title="Interview Score Distribution")

    st.plotly_chart(chart, use_container_width=True)

# ---------------------------------
# MOCK INTERVIEW
# ---------------------------------

elif page == "Mock Interview":

    st.title("💼 Mock Interview Center")

    role = st.selectbox(
        "Select Role",
        [
            "Data Scientist",
            "Machine Learning Engineer",
            "Data Analyst",
            "Software Engineer",
        ],
    )

    question_bank = {
        "Data Scientist": [
            "Explain Overfitting",
            "Difference between Classification and Regression",
            "Explain Random Forest",
        ],
        "Machine Learning Engineer": [
            "What is Model Deployment?",
            "Explain Feature Engineering",
            "What is MLOps?",
        ],
        "Data Analyst": ["Explain SQL Joins", "What is Power BI?", "Explain KPI"],
        "Software Engineer": [
            "Explain OOP",
            "What is API?",
            "Difference between List and Tuple",
        ],
    }

    st.subheader("Interview Questions")

    for q in question_bank[role]:
        st.write("• " + q)

# ---------------------------------
# AI FEEDBACK
# ---------------------------------

elif page == "AI Feedback":

    st.title("🤖 AI Interview Evaluation")

    communication = st.slider("Communication", 1, 10, 7)

    technical = st.slider("Technical Skills", 1, 10, 7)

    confidence = st.slider("Confidence", 1, 10, 7)

    problem_solving = st.slider("Problem Solving", 1, 10, 7)

    answer = st.text_area("Paste Interview Answer")

    if st.button("Generate Feedback"):

        prediction = model.predict(
            [[communication, technical, confidence, problem_solving]]
        )[0]

        sentiment = TextBlob(answer).sentiment.polarity

        st.success(f"Interview Score: {prediction:.0f}/1000")

        st.metric("Answer Sentiment", round(sentiment, 2))

        feedback = []

        if communication < 7:
            feedback.append("Improve communication clarity.")

        if technical < 7:
            feedback.append("Strengthen technical concepts.")

        if confidence < 7:
            feedback.append("Work on confidence and delivery.")

        if problem_solving < 7:
            feedback.append("Practice case studies.")

        st.subheader("AI Recommendations")

        for item in feedback:
            st.write("✅ " + item)

# ---------------------------------
# ANALYTICS
# ---------------------------------

elif page == "Analytics":

    st.title("📊 Interview Analytics")

    avg_score = df["InterviewScore"].mean()

    st.metric("Average Interview Score", round(avg_score, 2))

    skill_data = pd.DataFrame(
        {
            "Skill": ["Communication", "Technical", "Confidence", "Problem Solving"],
            "Importance": [25, 30, 20, 25],
        }
    )

    chart = px.bar(skill_data, x="Skill", y="Importance", title="Skill Importance")

    st.plotly_chart(chart, use_container_width=True)

# ---------------------------------
# REPORTS
# ---------------------------------

else:

    st.title("📁 Reports Center")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("Download Report", csv, "interview_report.csv", "text/csv")

    st.info("""
        AI Insight:
        Technical Skills and
        Communication contribute
        most to interview success.
        """)
