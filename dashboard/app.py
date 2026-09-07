"""
SupportSense NLP - Enterprise Customer Support Triage & SLA Routing Intelligence Platform
Future Interns Machine Learning Internship — Task 2
Dual-head NLP classification for real-time ticket categorization, priority scoring, calibrated confidence estimation, and operational SLA dispatch.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_ticket_text
from src.routing_engine import route_ticket

# Page Configuration
st.set_page_config(
    page_title="SupportSense NLP — Ticket Intelligence Platform",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Analytics Styling (Exact Palette Specification)
# Palette:
# Main Background: #F1F3EE | Sidebar: #E5EAE3 | Cards: #FFFFFF | Primary Teal: #087F7B | Dark Teal: #075E5B
# Primary Text: #1F2933 | Secondary Text: #53636A | High/Urgent: #D95D39 | Medium: #D99A24
# Low/Success: #5B8C72 | Borders: #D5DDD6 | Light Teal: #E7F4F2 | Light Urgent: #FBEAE5
# Light Medium: #FFF4D8 | Light Success: #EAF3ED
st.markdown("""
<style>
    /* Minimize Default Streamlit Header Chrome */
    header[data-testid="stHeader"] {
        background-color: rgba(241, 243, 238, 0.9) !important;
        height: 2.5rem !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Global Canvas */
    .stApp {
        background-color: #F1F3EE !important;
        color: #1F2933 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #E5EAE3 !important;
        border-right: 1px solid #D5DDD6 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #1F2933 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: #53636A !important;
    }

    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #1F2933 !important;
    }

    section[data-testid="stSidebar"] label p {
        color: #1F2933 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] code {
        background-color: #FFFFFF !important;
        color: #075E5B !important;
        border: 1px solid #D5DDD6 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
    }

    /* Structured Section Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        margin-bottom: 16px;
    }

    /* Section Number Badges */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 22px;
        margin-bottom: 12px;
    }
    
    .section-num {
        background-color: #087F7B;
        color: #FFFFFF;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1F2933;
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* Visual Workflow Steps */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04);
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .workflow-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        min-width: 110px;
    }
    
    .workflow-step-num {
        font-size: 0.70rem;
        font-weight: 800;
        color: #087F7B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .workflow-step-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1F2933;
        margin-top: 2px;
    }
    
    .workflow-arrow {
        color: #087F7B;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card-label {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #53636A;
        margin-bottom: 6px;
    }
    
    .metric-card-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1F2933;
        line-height: 1.25;
        word-break: break-word;
    }
    
    .metric-card-sub {
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid #F1F3EE;
    }

    /* Status Badges */
    .badge-urgent {
        background-color: #FBEAE5;
        color: #D95D39;
        border: 1px solid #F4C7BA;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-medium {
        background-color: #FFF4D8;
        color: #D99A24;
        border: 1px solid #F7DE98;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #EAF3ED;
        color: #5B8C72;
        border: 1px solid #BFDEC7;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-teal {
        background-color: #E7F4F2;
        color: #075E5B;
        border: 1px solid #B8E2DC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }

    .badge-neutral {
        background-color: #E5EAE3;
        color: #1F2933;
        border: 1px solid #D5DDD6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    
    .badge-channel {
        background-color: #F1F3EE;
        color: #1F2933;
        border: 1px solid #D5DDD6;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Textarea & Input Controls */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1F2933 !important;
        border: 1px solid #D5DDD6 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #087F7B !important;
        box-shadow: 0 0 0 1px #087F7B !important;
    }
    
    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #1F2933 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Primary Brand Button */
    .stButton button[kind="primary"] {
        background-color: #087F7B !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #075E5B !important;
    }
    
    /* Tabs & Data Tables */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #D5DDD6;
    }

    .stTabs [data-baseweb="tab"] {
        color: #53636A !important;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #087F7B !important;
        border-bottom: 2px solid #087F7B !important;
        font-weight: 700 !important;
    }

    /* Custom Light Benchmark Table Styling */
    .benchmark-table-container {
        background: #FFFFFF;
        border: 1px solid #D5DDD6;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.88rem;
    }
    
    .benchmark-table th {
        background-color: #F1F3EE;
        color: #1F2933;
        font-weight: 700;
        padding: 10px 14px;
        border-bottom: 1px solid #D5DDD6;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .benchmark-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F1F3EE;
        color: #1F2933;
    }
    
    .benchmark-table tr:last-child td {
        border-bottom: none;
    }
    
    .benchmark-table tr.champion-row td {
        background-color: #E7F4F2 !important;
        color: #075E5B !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Data & Artifact Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "support_sense_pipeline.pkl")
cat_val_metrics_path = os.path.join(base_dir, "outputs", "metrics", "category_validation_benchmarks.csv")
prio_val_metrics_path = os.path.join(base_dir, "outputs", "metrics", "priority_validation_benchmarks.csv")
cat_test_metrics_path = os.path.join(base_dir, "outputs", "metrics", "category_test_evaluation.csv")
prio_test_metrics_path = os.path.join(base_dir, "outputs", "metrics", "priority_test_evaluation.csv")

@st.cache_resource
def load_pipeline():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline_data = load_pipeline()

# Product Header Section
st.markdown("""
<div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #D5DDD6;">
    <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
        <h1 style="font-size: 2.1rem; font-weight: 900; color: #075E5B; margin: 0; letter-spacing: -0.03em;">
            SUPPORTSENSE
        </h1>
        <span style="font-size: 1.15rem; font-weight: 700; color: #1F2933; letter-spacing: -0.01em;">
            NLP TICKET INTELLIGENCE PLATFORM
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #53636A; font-weight: 500; margin-top: 4px; margin-bottom: 10px;">
        Automated customer support classification, priority scoring & operational routing
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-teal">NLP CLASSIFICATION</span>
        <span class="badge-teal">DUAL-HEAD MODEL</span>
        <span class="badge-teal">CALIBRATED CONFIDENCE</span>
        <span class="badge-neutral">70 / 15 / 15 VALIDATION</span>
        <span class="badge-neutral">ZERO TEST LEAKAGE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Visual Workflow Pipeline Section
st.markdown("""
<div class="workflow-container">
    <div class="workflow-step">
        <span class="workflow-step-num">Step 1</span>
        <span class="workflow-step-name">Incoming Ticket</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 2</span>
        <span class="workflow-step-name">NLP Processing</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 3</span>
        <span class="workflow-step-name">Category + Priority</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 4</span>
        <span class="workflow-step-name">Platt Confidence</span>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
        <span class="workflow-step-num">Step 5</span>
        <span class="workflow-step-name">Route & SLA</span>
    </div>
</div>
""", unsafe_allow_html=True)

if pipeline_data is None:
    st.error("⚠️ Model artifacts not found. Please run `python src/pipeline.py` to train and serialize the champion pipeline.")
    st.stop()

vectorizer = pipeline_data["vectorizer"]
cat_model = pipeline_data["category_model"]
prio_model = pipeline_data["priority_model"]
cat_champion_name = pipeline_data.get("category_model_name", "Multinomial Naive Bayes")
prio_champion_name = pipeline_data.get("priority_model_name", "Logistic Regression")

# Sidebar Configuration (Simplified & Focused on Product Functionality)
st.sidebar.markdown("""
<div style="padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid #D5DDD6;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #075E5B; letter-spacing: -0.02em;">
        SUPPORTSENSE
    </div>
    <div style="font-size: 0.82rem; font-weight: 600; color: #53636A;">
        NLP Ticket Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### Test Scenario")

sample_tickets = {
    "Payment Double Charge": "Credit card was charged twice for annual enterprise plan total $4,800. Please reverse erroneous charge immediately.",
    "Outage / Critical Crash": "Production Kubernetes cluster pods crashing in CrashLoopBackOff with OOMKilled errors affecting all EU users.",
    "Account Admin Lockout": "Organization admin locked out completely root credentials MFA authenticator lost urgent access required.",
    "Feature Request / Docs": "Minor typo in user documentation version 2.4 points to outdated 404 page in developer portal.",
    "Subscription Cancellation": "Cancel subscription immediately and process full refund of $2,200 as promised by account executive within 24 hours.",
    "Compliance / SOC2 Inquiry": "Need immediate confirmation if enterprise tier complies with HIPAA, SOC2 Type II, and GDPR requirements for contract signing.",
    "Custom Input": ""
}

preset_choice = st.sidebar.selectbox(
    "Choose a pre-filled ticket scenario:",
    options=list(sample_tickets.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Model")
st.sidebar.markdown(f"""
- **Category Champion:**  
  `{cat_champion_name}`
- **Priority Champion:**  
  `{prio_champion_name}`
- **Calibration:**  
  `Platt Scaling`
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Data Quality")
st.sidebar.markdown("""
- **✓ Zero exact duplicates**
- **✓ Zero normalized duplicates**
- **✓ Zero cross-split leakage**
- **✓ TF-IDF fitted on training data only**
""")

# Set Default Text Based on Selection
if preset_choice == "Custom Input":
    default_text = "Payment gateway timeout during checkout. Funds deducted from bank but our enterprise subscription is marked unpaid."
else:
    default_text = sample_tickets[preset_choice]

# Section 01: Incoming Support Ticket Workspace
st.markdown("""
<div class="section-header">
    <span class="section-num">01</span>
    <h3 class="section-title">Incoming Support Ticket</h3>
</div>
""", unsafe_allow_html=True)

user_ticket = st.text_area(
    "Customer Ticket Text / Inbound Message Body:",
    value=default_text,
    height=100,
    help="Enter customer email, chat transcript, or portal issue description."
)

col_btn, col_stats = st.columns([1, 4])
with col_btn:
    submit_btn = st.button("Classify & Dispatch", type="primary", use_container_width=True)
with col_stats:
    word_count = len(user_ticket.split()) if user_ticket else 0
    char_count = len(user_ticket) if user_ticket else 0
    st.markdown(f"""
    <div style="padding-top: 8px; display: flex; gap: 8px; align-items: center;">
        <span class="badge-neutral">Words: <strong style="color: #1F2933;">{word_count}</strong></span>
        <span class="badge-neutral">Characters: <strong style="color: #1F2933;">{char_count}</strong></span>
    </div>
    """, unsafe_allow_html=True)

# Run Inference
if user_ticket.strip():
    cleaned = clean_ticket_text(user_ticket)
    vec = vectorizer.transform([cleaned])
    
    cat_pred = cat_model.predict(vec)[0]
    prio_pred = prio_model.predict(vec)[0]
    
    cat_proba = cat_model.predict_proba(vec)[0] if hasattr(cat_model, "predict_proba") else np.array([1.0])
    prio_proba = prio_model.predict_proba(vec)[0] if hasattr(prio_model, "predict_proba") else np.array([1.0])
    
    cat_conf = float(np.max(cat_proba))
    prio_conf = float(np.max(prio_proba))
    joint_conf = float(cat_conf * prio_conf)
    
    routing = route_ticket(cat_pred, prio_pred, cat_conf, prio_conf)
    
    # Section 02: AI Triage Decision (HERO KPI Section)
    st.markdown("""
    <div class="section-header">
        <span class="section-num">02</span>
        <h3 class="section-title">AI Triage Decision</h3>
    </div>
    """, unsafe_allow_html=True)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Predicted Category</div>
                <div class="metric-card-value" style="color: #075E5B;">{cat_pred}</div>
            </div>
            <div class="metric-card-sub" style="color: #087F7B;">
                Category Conf: <strong>{cat_conf * 100:.1f}%</strong> (Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        if prio_pred == "High":
            prio_badge = '<span class="badge-urgent">HIGH</span>'
            prio_color = "#D95D39"
        elif prio_pred == "Medium":
            prio_badge = '<span class="badge-medium">MEDIUM</span>'
            prio_color = "#D99A24"
        else:
            prio_badge = '<span class="badge-success">LOW</span>'
            prio_color = "#5B8C72"
            
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Operational Priority</div>
                <div class="metric-card-value" style="margin-top: 4px;">{prio_badge}</div>
            </div>
            <div class="metric-card-sub" style="color: {prio_color};">
                Priority Conf: <strong>{prio_conf * 100:.1f}%</strong> (Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Joint Model Confidence</div>
                <div class="metric-card-value" style="color: #075E5B;">{joint_conf * 100:.1f}%</div>
            </div>
            <div class="metric-card-sub" style="color: #075E5B;">
                Joint Reliability: <strong>P(Cat) × P(Prio)</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        sla_hours = routing["SLA_Target_Hours"]
        sla_is_p1 = (sla_hours <= 2.0)
        sla_color = "#D95D39" if sla_is_p1 else "#5B8C72"
        sla_status_text = "Critical P1 Window" if sla_is_p1 else "Standard Window"
        
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">SLA Target</div>
                <div class="metric-card-value" style="color: {sla_color};">{sla_hours:.1f} Hours</div>
            </div>
            <div class="metric-card-sub" style="color: {sla_color};">
                Window: <strong>{sla_status_text}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section 03: Operational Routing Directive
    st.markdown("""
    <div class="section-header">
        <span class="section-num">03</span>
        <h3 class="section-title">Operational Routing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns([3, 2])
    
    with r_col1:
        is_escalated = routing["Auto_Escalation_Triggered"]
        esc_badge = '<span class="badge-urgent">Required (P1/P2)</span>' if is_escalated else '<span class="badge-success">Standard</span>'
        
        st.markdown(f"""
        <div class="content-box" style="margin-bottom: 0;">
            <div style="margin-bottom: 8px;"><strong style="color: #1F2933;">Operational Queue:</strong> <code style="background:#F1F3EE; color:#1F2933; border:1px solid #D5DDD6; padding:3px 8px; border-radius:6px; font-weight:700;">{routing['Assigned_Queue']}</code></div>
            <div style="margin-bottom: 8px;"><strong style="color: #1F2933;">Dispatch Channel:</strong> <span class="badge-channel">{routing['Channel']}</span> &nbsp; | &nbsp; <strong style="color: #1F2933;">Escalation:</strong> {esc_badge}</div>
            <div><strong style="color: #1F2933;">Routing Rationale:</strong> <span style="color: #53636A;">{routing['Routing_Rationale']}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with r_col2:
        if routing.get("Requires_Human_Triage", False) or joint_conf < 0.70:
            status_box_html = (
                '<div style="background-color: #FFF4D8; border: 1px solid #F7DE98; border-radius: 8px; padding: 12px 14px; color: #D99A24; font-size: 0.88rem; font-weight: 600;">'
                '⚠️ <strong>HUMAN VERIFICATION ADVISORY</strong><br>'
                '<span style="font-size: 0.82rem; font-weight: 500; color: #1F2933;">Confidence score is below the 70% threshold. Ticket flagged for supervisor review.</span>'
                '</div>'
            )
        else:
            status_box_html = (
                '<div style="background-color: #EAF3ED; border: 1px solid #BFDEC7; border-radius: 8px; padding: 12px 14px; color: #5B8C72; font-size: 0.88rem; font-weight: 600;">'
                '✓ <strong>AUTOMATED DISPATCH APPROVED</strong><br>'
                '<span style="font-size: 0.82rem; font-weight: 500; color: #1F2933;">Confidence exceeds safety guardrails. Automatic routing active.</span>'
                '</div>'
            )
        st.markdown(f"""
        <div class="content-box" style="margin-bottom: 0;">
            <div style="margin-bottom: 8px;"><strong style="color: #1F2933;">Dispatch Status:</strong></div>
            {status_box_html}
        </div>
        """, unsafe_allow_html=True)

    # Section 04: Model Confidence & Posterior Probabilities
    st.markdown("""
    <div class="section-header">
        <span class="section-num">04</span>
        <h3 class="section-title">Model Confidence</h3>
    </div>
    """, unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    plot_layout = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=40, t=35, b=20),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#53636A"),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor="#F1F3EE",
            zerolinecolor="#D5DDD6",
            tickfont=dict(size=10, color="#53636A")
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#1F2933", weight=600)
        )
    )
    
    with chart_col1:
        cat_classes = cat_model.classes_ if hasattr(cat_model, "classes_") else pipeline_data["category_labels"]
        cat_df = pd.DataFrame({"Category": cat_classes, "Probability": cat_proba}).sort_values(by="Probability", ascending=False)
        
        fig_cat = go.Figure(go.Bar(
            x=cat_df["Probability"],
            y=cat_df["Category"],
            orientation="h",
            marker=dict(
                color=["#075E5B" if c == cat_pred else "#D5DDD6" for c in cat_df["Category"]],
                line=dict(color=["#075E5B" if c == cat_pred else "#C2CCC3" for c in cat_df["Category"]], width=1)
            ),
            text=[f"{p * 100:.1f}%" for p in cat_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_cat.update_layout(
            title=dict(text="<b>Category Probability</b>", font=dict(size=13, color="#1F2933")),
            height=270,
            **plot_layout
        )
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})
        
    with chart_col2:
        prio_classes = prio_model.classes_ if hasattr(prio_model, "classes_") else pipeline_data["priority_labels"]
        prio_df = pd.DataFrame({"Priority": prio_classes, "Probability": prio_proba}).sort_values(by="Probability", ascending=False)
        
        prio_colors = {
            "High": "#D95D39",
            "Medium": "#D99A24",
            "Low": "#5B8C72"
        }
        
        fig_prio = go.Figure(go.Bar(
            x=prio_df["Probability"],
            y=prio_df["Priority"],
            orientation="h",
            marker=dict(
                color=[prio_colors.get(p, "#53636A") if p == prio_pred else "#D5DDD6" for p in prio_df["Priority"]],
                line=dict(color=[prio_colors.get(p, "#1F2933") for p in prio_df["Priority"]], width=1)
            ),
            text=[f"{p * 100:.1f}%" for p in prio_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_prio.update_layout(
            title=dict(text="<b>Priority Probability</b>", font=dict(size=13, color="#1F2933")),
            height=270,
            **plot_layout
        )
        st.plotly_chart(fig_prio, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("💡 Please enter a customer support ticket above or select a scenario from the sidebar, then click **Classify & Dispatch**.")

# Section 05: Empirical Model Benchmarks & Generalization Evaluation
st.markdown("""
<div class="section-header">
    <span class="section-num">05</span>
    <h3 class="section-title">Model Performance</h3>
</div>
<div style="color: #53636A; font-size: 0.90rem; margin-bottom: 12px;">
    Champions were selected using the 15% validation partition. The final 15% test partition was evaluated once after model selection.
</div>
""", unsafe_allow_html=True)

tab_cat, tab_prio, tab_test = st.tabs([
    "Category Benchmark (Validation)",
    "Priority Benchmark (Validation)",
    "Unbiased Final Test Evaluation"
])

def render_html_table(df, champion_name=None, champion_col="Model"):
    rows_html = []
    for _, row in df.iterrows():
        is_champ = (champion_name is not None and row.get(champion_col) == champion_name)
        row_class = ' class="champion-row"' if is_champ else ''
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                formatted_val = f"{val:.4f}"
            else:
                formatted_val = str(val)
                if is_champ and col == champion_col:
                    formatted_val = f"★ <strong>{formatted_val} (Champion)</strong>"
            cells.append(f"<td>{formatted_val}</td>")
        rows_html.append(f"<tr{row_class}>{''.join(cells)}</tr>")
    
    headers_html = "".join([f"<th>{col}</th>" for col in df.columns])
    
    return f"""
    <div class="benchmark-table-container">
        <table class="benchmark-table">
            <thead>
                <tr>{headers_html}</tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """

with tab_cat:
    if os.path.exists(cat_val_metrics_path):
        df_cat_val = pd.read_csv(cat_val_metrics_path)
        st.markdown(render_html_table(df_cat_val, champion_name=cat_champion_name), unsafe_allow_html=True)
        st.caption(f"★ **VALIDATION CHAMPION:** {cat_champion_name} (Selected strictly via 15% Validation Macro F1)")

with tab_prio:
    if os.path.exists(prio_val_metrics_path):
        df_prio_val = pd.read_csv(prio_val_metrics_path)
        st.markdown(render_html_table(df_prio_val, champion_name=prio_champion_name), unsafe_allow_html=True)
        st.caption(f"★ **VALIDATION CHAMPION:** {prio_champion_name} (Selected strictly via 15% Validation Macro F1)")

with tab_test:
    if os.path.exists(cat_test_metrics_path) and os.path.exists(prio_test_metrics_path):
        df_cat_test = pd.read_csv(cat_test_metrics_path)
        df_prio_test = pd.read_csv(prio_test_metrics_path)
        
        df_cat_test.insert(0, "Classification Head", "Head A: Ticket Category")
        df_prio_test.insert(0, "Classification Head", "Head B: Operational Priority")
        
        df_test_combined = pd.concat([df_cat_test, df_prio_test], ignore_index=True)
        st.markdown(render_html_table(df_test_combined), unsafe_allow_html=True)
        st.caption("Final evaluation conducted exactly once on the untouched 15% test partition (525 samples).")

# Section 06: Data Quality & Model Governance
st.markdown("""
<div class="section-header">
    <span class="section-num">06</span>
    <h3 class="section-title">Data Quality & Governance</h3>
</div>
""", unsafe_allow_html=True)

g_col1, g_col2, g_col3 = st.columns(3)
with g_col1:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-card-label">Training Partition</div>
        <div class="metric-card-value" style="color: #075E5B;">70%</div>
        <div class="metric-card-sub" style="color: #53636A;">2,450 Samples (TF-IDF Fitted)</div>
    </div>
    """, unsafe_allow_html=True)

with g_col2:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-card-label">Validation Partition</div>
        <div class="metric-card-value" style="color: #D99A24;">15%</div>
        <div class="metric-card-sub" style="color: #53636A;">525 Samples (Champion Selection)</div>
    </div>
    """, unsafe_allow_html=True)

with g_col3:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-card-label">Final Test Partition</div>
        <div class="metric-card-value" style="color: #5B8C72;">15%</div>
        <div class="metric-card-sub" style="color: #53636A;">525 Samples (Untouched Evaluation)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #FFFFFF; border: 1px solid #D5DDD6; border-radius: 14px; padding: 18px 22px; margin-top: 14px;">
    <h4 style="color: #1F2933; margin-top: 0; font-size: 0.96rem; font-weight: 700;">
        Dataset Provenance & Production Scope Notice
    </h4>
    <p style="font-size: 0.86rem; color: #53636A; line-height: 1.5; margin-bottom: 8px;">
        <strong style="color: #1F2933;">Dataset Nature:</strong> Curated multi-domain enterprise support ticket corpus (3,500 records) designed for evaluating multi-head NLP triage routing architectures. 
        Zero exact duplicates and zero cross-split leakage verified across all partitions.
    </p>
    <p style="font-size: 0.86rem; color: #53636A; line-height: 1.5; margin-bottom: 0;">
        <strong style="color: #1F2933;">Production Validation Notice:</strong> Because this is a curated synthetic prototype dataset, these results should not be interpreted as representative of production performance. Real-world performance should be validated on an independently collected, noisy production ticket dataset containing typos, slang, mixed intents, and distribution shift.
    </p>
</div>
""", unsafe_allow_html=True)
