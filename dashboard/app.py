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

# Custom Light Enterprise Analytics Styling
st.markdown("""
<style>
    /* Global Container and Typography */
    .stApp {
        background-color: #fbfaf7;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f5f3ec !important;
        border-right: 1px solid #e7e3da;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0f172a;
    }

    /* Clean Card Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e1d8;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    
    .metric-card-value {
        font-size: 1.22rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.3;
        word-break: break-word;
        white-space: normal;
    }
    
    .metric-card-sub {
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Structured Section Cards */
    .content-box {
        background: #ffffff;
        border: 1px solid #e5e1d8;
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    .content-box-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        border-bottom: 1px solid #f1eee7;
        padding-bottom: 8px;
    }

    /* Status Badges */
    .badge-urgent {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-medium {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-low {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-tag {
        background-color: #f8fafc;
        color: #334155;
        border: 1px solid #e2e8f0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-channel {
        background-color: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        padding: 2px 7px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Custom Input and Form Styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 1px #0f766e !important;
    }
    
    /* Selectbox readable text */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    
    /* Primary Action Button */
    .stButton button[kind="primary"] {
        background-color: #0f766e !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 24px !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #115e59 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #ffffff !important;
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
<div style="margin-bottom: 20px;">
    <h1 style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-bottom: 4px; letter-spacing: -0.02em;">
        SupportSense NLP
    </h1>
    <div style="font-size: 1.0rem; color: #475569; font-weight: 500; margin-bottom: 12px;">
        Enterprise Customer Support Ticket Triage & Dynamic SLA Routing Engine
    </div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-tag">Dual-Head NLP Architecture</span>
        <span class="badge-tag">TF-IDF + Platt Calibration</span>
        <span class="badge-tag">70% Train / 15% Val / 15% Test Split</span>
        <span class="badge-tag">Zero Test Leakage</span>
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
st.sidebar.markdown("### 📋 Preset Test Scenarios")
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
st.sidebar.markdown("### ⚙️ Engine Specifications")
st.sidebar.markdown(f"""
- **Category Champion:** `{cat_champion_name}`
- **Priority Champion:** `{prio_champion_name}`
- **Vector Space:** 2,631 n-grams (1, 2)
- **Calibration:** Platt Scaling (`CalibratedClassifierCV`)
- **Escalation Threshold:** Joint Confidence < 70%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
    <strong>Submission Details:</strong><br>
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
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.markdown('<div class="content-box-title">📥 Incoming Ticket Triage Workspace</div>', unsafe_allow_html=True)

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
    st.markdown(f"<div style='padding-top: 8px; color: #64748b; font-size: 0.85rem;'>Word Count: <strong>{word_count}</strong> | Character Count: <strong>{char_count}</strong></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown("### 🎯 Classification & Operational KPI Summary")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Predicted Category</div>
                <div class="metric-card-value">{cat_pred}</div>
            </div>
            <div class="metric-card-sub" style="color: #0369a1;">
                Confidence: <strong>{cat_conf * 100:.1f}%</strong> (Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        if prio_pred == "High":
            prio_badge = '<span class="badge-urgent">HIGH PRIORITY</span>'
            prio_sub_color = "#b91c1c"
        elif prio_pred == "Medium":
            prio_badge = '<span class="badge-medium">MEDIUM PRIORITY</span>'
            prio_sub_color = "#b45309"
        else:
            prio_badge = '<span class="badge-low">LOW PRIORITY</span>'
            prio_sub_color = "#15803d"
            
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
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">SLA Target Deadline</div>
                <div class="metric-card-value">{sla_hours:.1f} Hours</div>
            </div>
            <div class="metric-card-sub" style="color: {'#b91c1c' if sla_hours <= 2.0 else '#15803d'};">
                Status: <strong>{'Critical P1 Window' if sla_hours <= 2.0 else 'Standard Window'}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        is_escalated = routing["Auto_Escalation_Triggered"]
        esc_badge = '<span class="badge-urgent">ESCALATED (P1/P2)</span>' if is_escalated else '<span class="badge-low">STANDARD DISPATCH</span>'
        
        st.markdown(f"""
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Escalation Status</div>
                <div class="metric-card-value" style="margin-top: 4px;">{esc_badge}</div>
            </div>
            <div class="metric-card-sub" style="color: #475569;">
                Target: <span class="badge-channel">{routing['Channel']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section 3: Routing & SLA Decision Card
    st.markdown('<div class="content-box" style="margin-top: 10px;">', unsafe_allow_html=True)
    st.markdown('<div class="content-box-title">🧭 Automated Dispatch & Queue Directive</div>', unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns([3, 2])
    
    with r_col1:
        st.markdown(f"**Assigned Operational Queue:** `{routing['Assigned_Queue']}`")
        st.markdown(f"**Dispatch Channel / Pager Webhook:** `{routing['Channel']}`")
        st.markdown(f"**Operational Policy Rationale:** {routing['Routing_Rationale']}")
        
    with r_col2:
        st.markdown(f"**Joint Model Confidence Score:** `{joint_conf * 100:.1f}%`")
        if routing.get("Requires_Human_Triage", False) or joint_conf < 0.70:
            st.markdown("""
            <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 10px 14px; color: #92400e; font-size: 0.85rem;">
                ⚠️ <strong>Human Verification Advisory:</strong> Confidence score is near/below the safety threshold. Ticket has been flagged for supervisor review.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 10px 14px; color: #166534; font-size: 0.85rem;">
                ✓ <strong>Automated Dispatch Approved:</strong> Confidence exceeds safety guardrails. Ticket automatically routed to queue.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Probability Analysis
    st.markdown("### 📊 Calibrated Probability Distributions")
    chart_col1, chart_col2 = st.columns(2)
    
    # Plotly Theme Defaults
    plot_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=10, r=30, t=35, b=20),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#334155"),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#0f172a")
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
                color=["#0f766e" if c == cat_pred else "#94a3b8" for c in cat_df["Category"]],
                line=dict(color="#0f172a", width=0.5)
            ),
            text=[f"{p * 100:.1f}%" for p in cat_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_cat.update_layout(
            title=dict(text="<b>Category Posterior Probability</b>", font=dict(size=13, color="#0f172a")),
            height=280,
            **plot_layout
        )
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})
        
    with chart_col2:
        prio_classes = prio_model.classes_ if hasattr(prio_model, "classes_") else pipeline_data["priority_labels"]
        prio_df = pd.DataFrame({"Priority": prio_classes, "Probability": prio_proba}).sort_values(by="Probability", ascending=False)
        
        # Color mapping for priorities
        prio_colors = {
            "High": "#c2410c",
            "Medium": "#d97706",
            "Low": "#15803d"
        }
        
        fig_prio = go.Figure(go.Bar(
            x=prio_df["Probability"],
            y=prio_df["Priority"],
            orientation="h",
            marker=dict(
                color=[prio_colors.get(p, "#64748b") if p == prio_pred else "#cbd5e1" for p in prio_df["Priority"]],
                line=dict(color="#0f172a", width=0.5)
            ),
            text=[f"{p * 100:.1f}%" for p in prio_df["Probability"]],
            textposition="outside",
            cliponaxis=False
        ))
        
        fig_prio.update_layout(
            title=dict(text="<b>Priority Posterior Probability</b>", font=dict(size=13, color="#0f172a")),
            height=280,
            **plot_layout
        )
        st.plotly_chart(fig_prio, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("💡 Please enter a customer support ticket above or pick a scenario from the sidebar, then click **Classify & Dispatch**.")

# Section 5: Model Benchmark Comparisons
st.markdown("---")
st.markdown("### 🏆 Empirical Model Benchmarks & Generalization Evaluation")
st.markdown("Candidate models were trained strictly on the 70% training set and benchmarked on the 15% validation partition. Champions were selected **strictly on Validation Macro F1** and evaluated once on the untouched 15% test set.")

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
                return ["background-color: #f0fdf4; font-weight: bold; color: #166534;"] * len(row)
            return [""] * len(row)
            
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
        st.caption(f"★ Champion Model: **{cat_champion_name}** (Selected via Validation Macro F1)")

with tab_prio:
    if os.path.exists(prio_val_metrics_path):
        df_prio_val = pd.read_csv(prio_val_metrics_path)
        
        def highlight_prio_champion(row):
            if row["Model"] == prio_champion_name:
                return ["background-color: #f0fdf4; font-weight: bold; color: #166534;"] * len(row)
            return [""] * len(row)
            
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
        st.caption(f"★ Champion Model: **{prio_champion_name}** (Selected via Validation Macro F1)")

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
<div style="background-color: #ffffff; border: 1px solid #e5e1d8; border-radius: 8px; padding: 18px 22px; margin-top: 10px;">
    <h4 style="color: #0f172a; margin-top: 0; font-size: 0.95rem; font-weight: 700;">
        📌 Dataset Provenance & Production Scope Notice
    </h4>
    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5; margin-bottom: 8px;">
        <strong>Dataset Nature:</strong> Curated multi-domain enterprise support ticket corpus (3,500 records) designed for evaluating multi-head NLP triage routing architectures. 
        Zero exact duplicates and zero cross-split leakage verified across all partitions.
    </p>
    <p style="font-size: 0.86rem; color: #475569; line-height: 1.5; margin-bottom: 0;">
        <strong>Production Validation Notice:</strong> Because this is a curated synthetic prototype dataset, these results should not be interpreted as representative of production performance. Real-world performance should be validated on an independently collected, noisy production ticket dataset containing typos, slang, mixed intents, and distribution shift.
    </p>
</div>
""", unsafe_allow_html=True)

