"""
SupportSense NLP - Interactive Support Ticket Triage & Routing Dashboard
Streamlit web application for real-time ticket classification, priority tagging,
and automated SLA dispatch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_ticket_text
from src.routing_engine import route_ticket

st.set_page_config(page_title="SupportSense NLP", page_icon="🎫", layout="wide")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "support_sense_pipeline.pkl")
cat_metrics_path = os.path.join(base_dir, "outputs", "metrics", "category_model_benchmarks.csv")
prio_metrics_path = os.path.join(base_dir, "outputs", "metrics", "priority_model_benchmarks.csv")

@st.cache_resource
def load_pipeline():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline_data = load_pipeline()

# Title
st.title("🎫 SupportSense NLP — Intelligent Ticket Classification & Routing")
st.markdown("Automated dual-head NLP classification system for customer support tickets: Category prediction, priority assessment, confidence scoring, and dynamic queue dispatch.")

if pipeline_data is None:
    st.error("Model artifacts not found. Please run `python src/pipeline.py` first.")
    st.stop()

vectorizer = pipeline_data["vectorizer"]
cat_model = pipeline_data["category_model"]
prio_model = pipeline_data["priority_model"]

# Sidebar Presets
st.sidebar.header("📝 Preset Test Tickets")
sample_tickets = {
    "Custom Input": "",
    "Outage / Critical Crash": "Production database connection failure error code 500 server outage critical system down",
    "Billing Double Charge": "Credit card was double charged twice for annual subscription total $2,400 please reverse immediately",
    "Account Locked": "Account locked out completely admin cannot access production console MFA token expired",
    "Feature Request / Docs": "Typo in user settings documentation link points to older version 2.1 guide",
    "Subscription Cancellation": "Cancel subscription immediately and process full refund as promised by account manager within 24h",
    "Compliance / SOC2 Inquiry": "Does your enterprise plan meet HIPAA and SOC2 Type II compliance regulations for healthcare data?"
}

preset_choice = st.sidebar.selectbox("Choose a pre-filled ticket scenario:", list(sample_tickets.keys()))

default_text = sample_tickets[preset_choice] if preset_choice != "Custom Input" else "Payment failed twice and my account is locked"

# Ticket Input
st.subheader("📥 Incoming Ticket Triage Workspace")
user_ticket = st.text_area("Customer Ticket Text / Email Body:", value=default_text, height=120)

if st.button("🚀 Classify & Dispatch Ticket", type="primary"):
    if not user_ticket.strip():
        st.warning("Please enter some ticket text.")
    else:
        cleaned = clean_ticket_text(user_ticket)
        vec = vectorizer.transform([cleaned])
        
        cat_pred = cat_model.predict(vec)[0]
        prio_pred = prio_model.predict(vec)[0]
        
        cat_proba = cat_model.predict_proba(vec)[0] if hasattr(cat_model, "predict_proba") else [1.0]
        prio_proba = prio_model.predict_proba(vec)[0] if hasattr(prio_model, "predict_proba") else [1.0]
        
        cat_conf = np.max(cat_proba)
        prio_conf = np.max(prio_proba)
        
        routing = route_ticket(cat_pred, prio_pred, cat_conf, prio_conf)
        
        # Display Prediction Cards
        st.divider()
        st.subheader("🎯 Automated Triage & Dispatch Decision")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Category", cat_pred, f"{cat_conf*100:.1f}% Confidence")
        
        prio_color = "🔴" if prio_pred == "High" else ("🟡" if prio_pred == "Medium" else "🟢")
        col2.metric("Predicted Priority", f"{prio_color} {prio_pred}", f"{prio_conf*100:.1f}% Confidence")
        
        col3.metric("SLA Resolution Target", f"{routing['SLA_Target_Hours']} Hours", "Escalated" if routing["Auto_Escalation_Triggered"] else "Standard")
        col4.metric("Assigned Support Queue", routing["Assigned_Queue"].split()[0] + " Desk", routing["Channel"])

        # Detailed Routing Breakdown
        st.info(f"**Assigned Support Queue:** `{routing['Assigned_Queue']}` | **Channel:** `{routing['Channel']}`\n\n**Routing Rationale:** {routing['Routing_Rationale']}")
        
        if routing["Requires_Human_Triage"]:
            st.warning("⚠️ **Low Confidence Flag:** Joint confidence score is below safe automation threshold (< 60%). Routed to human triage pool for confirmation.")

        # Class Probability Distributions
        st.divider()
        st.subheader("📊 Model Prediction Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            cat_classes = cat_model.classes_ if hasattr(cat_model, "classes_") else pipeline_data["category_labels"]
            cat_df = pd.DataFrame({"Category": cat_classes, "Probability": cat_proba}).sort_values(by="Probability", ascending=False)
            fig_cat = px.bar(cat_df, x="Probability", y="Category", orientation="h", title="Category Probability Distribution", color="Probability", color_continuous_scale="Blues")
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with prob_col2:
            prio_classes = prio_model.classes_ if hasattr(prio_model, "classes_") else pipeline_data["priority_labels"]
            prio_df = pd.DataFrame({"Priority": prio_classes, "Probability": prio_proba}).sort_values(by="Probability", ascending=False)
            fig_prio = px.bar(prio_df, x="Probability", y="Priority", orientation="h", title="Priority Probability Distribution", color="Probability", color_continuous_scale="Reds")
            st.plotly_chart(fig_prio, use_container_width=True)

# Benchmarks Tab
st.divider()
st.subheader("🏆 Model Benchmark Comparisons")
bcol1, bcol2 = st.columns(2)
with bcol1:
    if os.path.exists(cat_metrics_path):
        st.markdown("**Category Classifier Benchmark:**")
        st.dataframe(pd.read_csv(cat_metrics_path), use_container_width=True)
with bcol2:
    if os.path.exists(prio_metrics_path):
        st.markdown("**Priority Classifier Benchmark:**")
        st.dataframe(pd.read_csv(prio_metrics_path), use_container_width=True)
