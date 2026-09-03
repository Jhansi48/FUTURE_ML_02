"""
SupportSense NLP - Support Queue Routing & SLA Recommendation Engine
Translates joint Category + Priority predictions into operational support queues,
SLA resolution deadlines, and human verification triggers.
"""

from typing import Dict, Any

ROUTING_POLICY = {
    ("Technical Issues", "High"): {
        "Queue": "Tier-3 Site Reliability & Core Engineering",
        "SLA_Hours": 1.0,
        "Auto_Escalate": True,
        "Notification_Channel": "#ops-sev1-critical",
        "Description": "Critical production outage, data pipeline failure, or security emergency."
    },
    ("Technical Issues", "Medium"): {
        "Queue": "Tier-2 Application Support Desk",
        "SLA_Hours": 6.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#tier2-support",
        "Description": "Application latency, minor webhook delays, or localized functional glitches."
    },
    ("Technical Issues", "Low"): {
        "Queue": "Tier-1 Helpdesk / Developer Community",
        "SLA_Hours": 24.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#community-support",
        "Description": "Cosmetic UI requests, documentation typos, or client-side caching queries."
    },
    ("Billing & Payments", "High"): {
        "Queue": "Priority Financial Operations & Merchant Desk",
        "SLA_Hours": 2.0,
        "Auto_Escalate": True,
        "Notification_Channel": "#billing-urgent",
        "Description": "Double charges, erroneous account suspension, or payment gateway blockage."
    },
    ("Billing & Payments", "Medium"): {
        "Queue": "Customer Billing & Invoicing Desk",
        "SLA_Hours": 8.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#billing-general",
        "Description": "Invoice tax updates, ACH conversion requests, or usage quota breakdowns."
    },
    ("Billing & Payments", "Low"): {
        "Queue": "Self-Service Billing Portal / Automated FAQ",
        "SLA_Hours": 24.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#billing-selfservice",
        "Description": "Invoice downloads, currency support queries, or renewal dates."
    },
    ("Account Access", "High"): {
        "Queue": "Identity, Access Management & Security Desk",
        "SLA_Hours": 1.0,
        "Auto_Escalate": True,
        "Notification_Channel": "#sec-iam-urgent",
        "Description": "Admin lockout, SAML SSO enterprise failure, or suspected account takeover."
    },
    ("Account Access", "Medium"): {
        "Queue": "User Provisioning & Account Support",
        "SLA_Hours": 4.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#iam-support",
        "Description": "Password reset token issues, team invitation errors, or email modifications."
    },
    ("Account Access", "Low"): {
        "Queue": "Self-Service User Guide & Chatbot",
        "SLA_Hours": 24.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#general-support",
        "Description": "Profile avatar customization or authenticator app setup guides."
    },
    ("Product Inquiries", "High"): {
        "Queue": "Enterprise Solutions Architecture & Presales",
        "SLA_Hours": 3.0,
        "Auto_Escalate": True,
        "Notification_Channel": "#enterprise-deals",
        "Description": "Urgent compliance questionnaires (SOC2/HIPAA) or closing contract queries."
    },
    ("Product Inquiries", "Medium"): {
        "Queue": "Customer Success & Product Specialists",
        "SLA_Hours": 12.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#product-support",
        "Description": "Plan tier comparison, scheduled export integrations, or database connectors."
    },
    ("Product Inquiries", "Low"): {
        "Queue": "Knowledge Base / Automated AI Concierge",
        "SLA_Hours": 48.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#community",
        "Description": "General feature roadmap queries, webinar recordings, or tutorial links."
    },
    ("Cancellation & Refunds", "High"): {
        "Queue": "Executive Escalations & Retention Taskforce",
        "SLA_Hours": 2.0,
        "Auto_Escalate": True,
        "Notification_Channel": "#retention-urgent",
        "Description": "Urgent charge dispute, SLA breach termination, or immediate refund demand."
    },
    ("Cancellation & Refunds", "Medium"): {
        "Queue": "Account Management & Commercial Desk",
        "SLA_Hours": 8.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#commercial-ops",
        "Description": "Plan downgrade, addon cancellation, or partial license credit request."
    },
    ("Cancellation & Refunds", "Low"): {
        "Queue": "Standard Customer Service Desk",
        "SLA_Hours": 24.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#customer-service",
        "Description": "Cancellation timeline queries or account pause policies."
    }
}

def route_ticket(category: str, priority: str, cat_conf: float, prio_conf: float) -> Dict[str, Any]:
    """Generates complete routing directive with SLA and confidence check."""
    key = (category, priority)
    policy = ROUTING_POLICY.get(key, {
        "Queue": "General Support Triage",
        "SLA_Hours": 24.0,
        "Auto_Escalate": False,
        "Notification_Channel": "#triage-pool",
        "Description": "Unclassified ticket requiring manual triage."
    })
    
    # Confidence guardrail (Threshold: 0.60)
    requires_human_verification = bool(cat_conf < 0.60 or prio_conf < 0.60)
    
    return {
        "Assigned_Queue": policy["Queue"],
        "SLA_Target_Hours": policy["SLA_Hours"],
        "Auto_Escalation_Triggered": policy["Auto_Escalate"],
        "Channel": policy["Notification_Channel"],
        "Routing_Rationale": policy["Description"],
        "Requires_Human_Triage": requires_human_verification,
        "Confidence_Score": {
            "Category_Confidence": round(float(cat_conf), 4),
            "Priority_Confidence": round(float(prio_conf), 4),
            "Joint_Confidence": round(float(cat_conf * prio_conf), 4)
        }
    }
