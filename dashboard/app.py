"""
SupportSense NLP - Enterprise Customer Support Triage & SLA Routing Dashboard
A high-performance dual-head NLP classification system for real-time ticket categorization,
calibrated priority assessment, deterministic SLA dispatch, and incident escalation.
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
    page_title="SupportSense NLP — Ticket Triage & Routing",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Analytics Styling (Exact Palette Specification)
# Palette:
# Background: #F8F7F3 | Cards: #FFFFFF | Primary Brand: #087F7B | Primary Dark: #075E5B
# Main Text: #1F2933 | Secondary Text: #5B6770 | High/Urgent: #D95D39 | Medium: #D99A24
# Low/Success: #5B8C72 | Borders: #E3E5E2 | Light Teal: #E7F4F2 | Light Urgent: #FBEAE5
# Light Medium: #FFF4D8 | Light Success: #EAF3ED
st.markdown("""
<style>
    /* Global App Canvas */
    .stApp {
        background-color: #F8F7F3 !important;
        color: #1F2933 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F2EFE9 !important;
        border-right: 1px solid #E3E5E2 !important;
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
        color: #5B6770 !important;
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
        border: 1px solid #E3E5E2 !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
    }

    /* Structured Section Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E3E5E2;
        border-radius: 8px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        margin-bottom: 20px;
    }
    
    .content-box-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1F2933;
        margin-bottom: 14px;
        border-bottom: 1px solid #E3E5E2;
        padding-bottom: 8px;
        letter-spacing: -0.01em;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E3E5E2;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(31, 41, 51, 0.04), 0 1px 2px rgba(31, 41, 51, 0.02);
        margin-bottom: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #5B6770;
        margin-bottom: 6px;
    }
    
    .metric-card-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1F2933;
        line-height: 1.3;
        word-break: break-word;
        white-space: normal;
    }
    
    .metric-card-sub {
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Status Badges */
    .badge-urgent {
        background-color: #FBEAE5;
        color: #D95D39;
        border: 1px solid #F4C7BA;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .badge-medium {
        background-color: #FFF4D8;
        color: #D99A24;
        border: 1px solid #F7DE98;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #EAF3ED;
        color: #5B8C72;
        border: 1px solid #BFDEC7;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .badge-teal {
        background-color: #E7F4F2;
        color: #075E5B;
        border: 1px solid #B8E2DC;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }

    .badge-neutral {
        background-color: #F2EFE9;
        color: #1F2933;
        border: 1px solid #E3E5E2;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .badge-channel {
        background-color: #F2EFE9;
        color: #1F2933;
        border: 1px solid #E3E5E2;
        padding: 2px 7px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Textarea & Input Controls */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1F2933 !important;
        border: 1px solid #E3E5E2 !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #087F7B !important;
        box-shadow: 0 0 0 1px #087F7B !important;
    }
    
    /* Selectbox Input Controls */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 6px !important;
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
        border-radius: 6px !important;
        padding: 8px 24px !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #075E5B !important;
    }
    
    /* Tabs & DataFrames */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E3E5E2;
    }

    .stTabs [data-baseweb="tab"] {
        color: #5B6770 !important;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #087F7B !important;
        border-bottom: 2px solid #087F7B !important;
        font-weight: 700 !important;
    }

    .stDataFrame {
        background-color: #FFFFFF !important;
        border: 1px solid #E3E5E2;
        border-radius: 6px;
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

# Header Section
st.markdown("""
<div style="margin-bottom: 22px;">
    <h1 style="font-size: 1.95rem; font-weight: 800; color: #1F2933; margin-bottom: 4px; letter-spacing: -0.02em;">
        SupportSense NLP
    </h1>
    <div style="font-size: 1.02rem; color: #5B6770; font-weight: 500; margin-bottom: 12px;">
        Enterprise Customer Support Ticket Triage & Dynamic SLA Routing Engine
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-teal">Dual-Head NLP Architecture</span>
        <span class="badge-neutral">TF-IDF + Platt Calibration</span>
        <span class="badge-neutral">70% Train / 15% Val / 15% Test Split</span>
        <span class="badge-success">Zero Test Leakage Verified</span>
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

# Sidebar Configuration
st.sidebar.markdown("### Preset Test Scenarios")
st.sidebar.markdown("Select a real-world enterprise support scenario to test model triage, calibrated confidence scoring, and routing directives.")

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
st.sidebar.markdown("### Engine Specifications")
st.sidebar.markdown(f"""
- **Category Champion:** `{cat_champion_name}`
- **Priority Champion:** `{prio_champion_name}`
- **Vector Space:** 2,631 n-grams (1, 2)
- **Calibration:** Platt Scaling (`CalibratedClassifierCV`)
- **Escalation Policy:** Joint Confidence < 70%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.80rem; color: #5B6770; line-height: 1.45;">
    <strong style="color: #1F2933;">Submission Metadata:</strong><br>
    Internship ID: <code>FIT/AUG26/ML10465</code><br>
    Track: Machine Learning (Task 2)<br>
    Repository: <code>FUTURE_ML_02</code>
</div>
""", unsafe_allow_html=True)

# Set Default Text Based on Selection
if preset_choice == "Custom Input":
    default_text = "Payment gateway timeout during checkout. Funds deducted from bank but our enterprise subscription is marked unpaid."
else:
    default_text = sample_tickets[preset_choice]

# Section 1: Incoming Ticket Workspace
st.markdown("### Incoming Ticket Triage Workspace")

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
    st.markdown(f"<div style='padding-top: 8px; color: #5B6770; font-size: 0.86rem;'>Word Count: <strong style='color: #1F2933;'>{word_count}</strong> | Character Count: <strong style='color: #1F2933;'>{char_count}</strong></div>", unsafe_allow_html=True)

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
    
    # Section 2: Triage & Classification KPI Overview
    st.markdown("### Classification & Operational KPI Summary")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Predicted Category</div>
                <div class="metric-card-value">{cat_pred}</div>
            </div>
            <div class="metric-card-sub" style="color: #087F7B;">
                Confidence: <strong>{cat_conf * 100:.1f}%</strong> (Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        if prio_pred == "High":
            prio_badge = '<span class="badge-urgent">HIGH PRIORITY</span>'
            prio_sub_color = "#D95D39"
        elif prio_pred == "Medium":
            prio_badge = '<span class="badge-medium">MEDIUM PRIORITY</span>'
            prio_sub_color = "#D99A24"
        else:
            prio_badge = '<span class="badge-success">LOW PRIORITY</span>'
            prio_sub_color = "#5B8C72"
            
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Operational Priority</div>
                <div class="metric-card-value" style="margin-top: 4px;">{prio_badge}</div>
            </div>
            <div class="metric-card-sub" style="color: {prio_sub_color};">
                Confidence: <strong>{prio_conf * 100:.1f}%</strong> (Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        sla_hours = routing["SLA_Target_Hours"]
        sla_is_p1 = (sla_hours <= 2.0)
        sla_color = "#D95D39" if sla_is_p1 else "#5B8C72"
        sla_status_text = "Critical P1 Window" if sla_is_p1 else "Standard Window"
        
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">SLA Target Deadline</div>
                <div class="metric-card-value">{sla_hours:.1f} Hours</div>
            </div>
            <div class="metric-card-sub" style="color: {sla_color};">
                Status: <strong>{sla_status_text}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        is_escalated = routing["Auto_Escalation_Triggered"]
        esc_badge = '<span class="badge-urgent">ESCALATED (P1/P2)</span>' if is_escalated else '<span class="badge-success">STANDARD DISPATCH</span>'
        
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Escalation Status</div>
                <div class="metric-card-value" style="margin-top: 4px;">{esc_badge}</div>
            </div>
            <div class="metric-card-sub" style="color: #5B6770;">
                Target: <span class="badge-channel">{routing['Channel']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section 3: Routing & SLA Decision Card
    st.markdown("### Automated Dispatch & Queue Directive")
    
    r_col1, r_col2 = st.columns([3, 2])
    
    with r_col1:
        st.markdown(f"""
        <div class="content-box" style="margin-bottom: 0;">
            <div style='margin-bottom: 8px;'><strong style='color: #1F2933;'>Assigned Operational Queue:</strong> <code style='background:#F2EFE9; color:#1F2933; border:1px solid #E3E5E2; padding:2px 6px; border-radius:4px;'>{routing['Assigned_Queue']}</code></div>
            <div style='margin-bottom: 8px;'><strong style='color: #1F2933;'>Dispatch Channel / Pager Webhook:</strong> <span class='badge-channel'>{routing['Channel']}</span></div>
            <div><strong style='color: #1F2933;'>Operational Policy Rationale:</strong> <span style='color: #5B6770;'>{routing['Routing_Rationale']}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with r_col2:
        if routing.get("Requires_Human_Triage", False) or joint_conf < 0.70:
            advisory_box = """
            <div style="background-color: #FFF4D8; border: 1px solid #F7DE98; border-radius: 6px; padding: 10px 14px; color: #D99A24; font-size: 0.86rem; font-weight: 600; margin-top: 8px;">
                ⚠️ Human Verification Advisory: Confidence score is below the 70% threshold. Ticket flagged for supervisor review.
            </div>
            """
        else:
            advisory_box = """
            <div style="background-color: #EAF3ED; border: 1px solid #BFDEC7; border-radius: 6px; padding: 10px 14px; color: #5B8C72; font-size: 0.86rem; font-weight: 600; margin-top: 8px;">
                ✓ Automated Dispatch Approved: Confidence exceeds safety guardrails. Automatic routing active.
            </div>
            """
        st.markdown(f"""
        <div class="content-box" style="margin-bottom: 0;">
            <div><strong style='color: #1F2933;'>Joint Model Confidence Score:</strong> <strong style='color:#087F7B;'>{joint_conf * 100:.1f}%</strong></div>
            {advisory_box}
        </div>
        """, unsafe_allow_html=True)

    # Section 4: Probability Analysis
    st.markdown("### Calibrated Probability Distributions")
    chart_col1, chart_col2 = st.columns(2)
    
    # Plotly Theme Defaults matching exact palette
    plot_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=35, t=35, b=20),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#5B6770"),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor="#F2EFE9",
            zerolinecolor="#E3E5E2",
            tickfont=dict(size=10, color="#5B6770")
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
                color=["#087F7B" if c == cat_pred else "#C2E5E2" for c in cat_df["Category"]],
                line=dict(color=["#075E5B" if c == cat_pred else "#E3E5E2" for c in cat_df["Category"]], width=1)
            ),
            text=[f"{p * 100:.1f}%" for p in cat_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_cat.update_layout(
            title=dict(text="<b>Category Posterior Probability</b>", font=dict(size=13, color="#1F2933")),
            height=280,
            **plot_layout
        )
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})
        
    with chart_col2:
        prio_classes = prio_model.classes_ if hasattr(prio_model, "classes_") else pipeline_data["priority_labels"]
        prio_df = pd.DataFrame({"Priority": prio_classes, "Probability": prio_proba}).sort_values(by="Probability", ascending=False)
        
        # Color mapping for priorities matching exact palette
        prio_colors = {
            "High": "#D95D39",
            "Medium": "#D99A24",
            "Low": "#5B8C72"
        }
        
        prio_muted = {
            "High": "#FBEAE5",
            "Medium": "#FFF4D8",
            "Low": "#EAF3ED"
        }
        
        fig_prio = go.Figure(go.Bar(
            x=prio_df["Probability"],
            y=prio_df["Priority"],
            orientation="h",
            marker=dict(
                color=[prio_colors.get(p, "#5B6770") if p == prio_pred else prio_muted.get(p, "#E3E5E2") for p in prio_df["Priority"]],
                line=dict(color=[prio_colors.get(p, "#1F2933") for p in prio_df["Priority"]], width=1)
            ),
            text=[f"{p * 100:.1f}%" for p in prio_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_prio.update_layout(
            title=dict(text="<b>Priority Posterior Probability</b>", font=dict(size=13, color="#1F2933")),
            height=280,
            **plot_layout
        )
        st.plotly_chart(fig_prio, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("💡 Please enter a customer support ticket above or pick a scenario from the sidebar, then click **Classify & Dispatch**.")

# Section 5: Model Benchmark Comparisons
st.markdown("---")
st.markdown("### Empirical Model Benchmarks & Generalization Evaluation")
st.markdown("<div style='color: #5B6770; font-size: 0.90rem; margin-bottom: 12px;'>Candidate models were trained on the 70% training partition and benchmarked on the 15% validation partition. Champions were selected <strong>strictly on Validation Macro F1</strong> and evaluated once on the untouched 15% test partition.</div>", unsafe_allow_html=True)

tab_cat, tab_prio, tab_test = st.tabs([
    "Head A: Category Benchmarks (Validation)",
    "Head B: Priority Benchmarks (Validation)",
    "Unbiased Final Test Evaluation (Champions)"
])

with tab_cat:
    if os.path.exists(cat_val_metrics_path):
        df_cat_val = pd.read_csv(cat_val_metrics_path)
        
        def highlight_cat_champion(row):
            if row["Model"] == cat_champion_name:
                return ["background-color: #E7F4F2; font-weight: bold; color: #075E5B;"] * len(row)
            return ["color: #1F2933;"] * len(row)
            
        st.dataframe(
            df_cat_val.style.format({
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}"
            }).apply(highlight_cat_champion, axis=1),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Selected Champion Model: **{cat_champion_name}** (Selected via 15% Validation Macro F1)")

with tab_prio:
    if os.path.exists(prio_val_metrics_path):
        df_prio_val = pd.read_csv(prio_val_metrics_path)
        
        def highlight_prio_champion(row):
            if row["Model"] == prio_champion_name:
                return ["background-color: #E7F4F2; font-weight: bold; color: #075E5B;"] * len(row)
            return ["color: #1F2933;"] * len(row)
            
        st.dataframe(
            df_prio_val.style.format({
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}"
            }).apply(highlight_prio_champion, axis=1),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Selected Champion Model: **{prio_champion_name}** (Selected via 15% Validation Macro F1)")

with tab_test:
    if os.path.exists(cat_test_metrics_path) and os.path.exists(prio_test_metrics_path):
        df_cat_test = pd.read_csv(cat_test_metrics_path)
        df_prio_test = pd.read_csv(prio_test_metrics_path)
        
        df_cat_test.insert(0, "Classification Head", "Head A: Ticket Category")
        df_prio_test.insert(0, "Classification Head", "Head B: Operational Priority")
        
        df_test_combined = pd.concat([df_cat_test, df_prio_test], ignore_index=True)
        st.dataframe(
            df_test_combined.style.format({
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Final evaluation conducted exactly once on the untouched 15% test partition (525 samples).")

# Section 6: Dataset Provenance & Production Scope
st.markdown("---")
st.markdown("""
<div style="background-color: #FFFFFF; border: 1px solid #E3E5E2; border-radius: 8px; padding: 18px 22px; margin-top: 10px;">
    <h4 style="color: #1F2933; margin-top: 0; font-size: 0.96rem; font-weight: 700;">
        Dataset Provenance & Production Scope Notice
    </h4>
    <p style="font-size: 0.86rem; color: #5B6770; line-height: 1.5; margin-bottom: 8px;">
        <strong style="color: #1F2933;">Dataset Nature:</strong> Curated multi-domain enterprise support ticket corpus (3,500 records) designed for evaluating multi-head NLP triage routing architectures. 
        Zero exact duplicates and zero cross-split leakage verified across all partitions.
    </p>
    <p style="font-size: 0.86rem; color: #5B6770; line-height: 1.5; margin-bottom: 0;">
        <strong style="color: #1F2933;">Production Validation Notice:</strong> Because this is a curated synthetic prototype dataset, these results should not be interpreted as representative of production performance. Real-world performance should be validated on an independently collected, noisy production ticket dataset containing typos, slang, mixed intents, and distribution shift.
    </p>
</div>
""", unsafe_allow_html=True)


