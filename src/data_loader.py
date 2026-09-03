"""
SupportSense NLP - Data Ingestion & Loading Module
Source: Customer Support Ticket Dataset (Enterprise benchmark across SaaS, E-Commerce, and FinTech)
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple

def generate_support_ticket_dataset(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic multi-class, multi-priority customer support ticket dataset
    with natural customer phrasing, domain jargon, urgency signals, and realistic noise.
    """
    np.random.seed(random_state)
    
    # 5 Key Categories with distinct intent patterns and sub-scenarios
    categories_data = {
        "Technical Issues": {
            "high": [
                "Production database connection failure error code 500 server outage critical system down",
                "API endpoint returning 503 service unavailable all background workers crashed immediately",
                "Security vulnerability detected unauthorized API calls from unknown IP address help",
                "Memory leak causing production Kubernetes pods to crash in loop OOMKilled",
                "Data pipeline corrupted our daily sync failed for enterprise client accounts"
            ],
            "medium": [
                "Web application loading very slowly dashboard takes over 30 seconds to render charts",
                "Cannot upload CSV file larger than 5MB error says file format unsupported",
                "Webhook notifications are delayed by approximately 45 minutes since latest release",
                "Mobile app crashes whenever user switches from dark mode to light mode in settings",
                "Integration with Slack bot stopped sending automated channel alerts"
            ],
            "low": [
                "Minor CSS layout glitch on safari browser when viewing navigation menu",
                "Typo in user settings documentation link points to older version 2.1 guide",
                "Feature request: can you add a dark theme option for the PDF export tool?",
                "How do I clear browser cache to view updated profile picture?",
                "Is there an RSS feed available for platform status announcements?"
            ]
        },
        "Billing & Payments": {
            "high": [
                "Credit card was double charged twice for annual subscription total $2,400 please reverse immediately",
                "Payment gateway failed during checkout customer funds deducted but invoice marked unpaid",
                "Account suspended due to false payment failure despite valid corporate credit card on file",
                "Fraudulent unauthorized transactions detected on our billing card linked to this account",
                "Urgent: tax invoice number missing on compliance statement audit deadline tomorrow"
            ],
            "medium": [
                "Need to update billing address and GST identification number on upcoming monthly invoice",
                "Coupon promo code DISCOUNT20 was not applied during renewal checkout",
                "Requesting breakdown of usage-based tier charges for bandwidth overages last month",
                "How to switch payment method from American Express to corporate ACH wire transfer?",
                "Payment receipt not delivered to finance accounting team email address"
            ],
            "low": [
                "Where can I download past invoices from the billing settings page?",
                "Will our annual subscription automatically renew next quarter?",
                "Requesting quote for adding 5 additional user seats to team plan next month",
                "What payment currencies do you support for international bank cards?",
                "Can we pay our invoice via annual bank check instead of card?"
            ]
        },
        "Account Access": {
            "high": [
                "Account locked out completely admin cannot access production console MFA token expired",
                "Suspected account takeover received password reset email that was not initiated by us",
                "SSO SAML authentication broken for entire organization no employees can log in",
                "Lost two-factor authentication device backup codes not working urgent root access needed",
                "Admin account deleted by mistake restore access and permissions urgently"
            ],
            "medium": [
                "Password reset link email never arrives in inbox or spam folder",
                "Cannot change primary email address in profile settings page throws permission error",
                "Need to revoke access for former employee immediately and transfer project ownership",
                "Google workspace OAuth single sign-on throws redirect uri mismatch error",
                "Invited team member receives invalid invitation token error when clicking join link"
            ],
            "low": [
                "How do I update my profile avatar and display name on the portal?",
                "Can one user be a member of multiple organization workspaces?",
                "Where do I configure two-factor authentication in account preferences?",
                "How to change notification preferences for email digests?",
                "Requesting instructions for setting up Google Authenticator app"
            ]
        },
        "Product Inquiries": {
            "high": [
                "Does your enterprise plan meet HIPAA and SOC2 Type II compliance regulations for healthcare data?",
                "Need immediate technical specifications before signing annual enterprise contract today",
                "Critical compatibility question: does your SDK support Python 3.12 in production?",
                "Urgent clarification on API rate limits for upcoming marketing campaign launching tonight",
                "Data residency inquiry: can our customer data be strictly stored in Frankfurt EU region?"
            ],
            "medium": [
                "What is the exact difference in features between Professional and Enterprise tier?",
                "Does the platform support automated CSV data export via scheduled SFTP?",
                "Can we customize webhook payload headers with custom authorization tokens?",
                "What are the supported database connectors for the analytics pipeline?",
                "How many concurrent API requests are allowed on the standard developer tier?"
            ],
            "low": [
                "Is there a free trial period available for the analytics add-on module?",
                "Where can I find video tutorials for onboarding new team members?",
                "Do you have a public product roadmap for upcoming Q3 features?",
                "Can you share best practices for organizing workspaces across departments?",
                "Where is the latest API documentation and Postman collection available?"
            ]
        },
        "Cancellation & Refunds": {
            "high": [
                "Cancel subscription immediately and process full refund as promised by account manager within 24h",
                "Subscription renewed without 30-day notice demanding immediate full refund of $1,800",
                "Service SLA breach below 99% uptime requesting contract termination and penalty refund",
                "Customer onboarding failed completely demanding full refund and data wipe immediately",
                "Legal notice: unauthorized renewal charge dispute initiated process refund now"
            ],
            "medium": [
                "We are downsizing our team and wish to downgrade from Enterprise to Starter plan",
                "Please cancel our monthly recurring addon package before next billing cycle begins",
                "Customer requested partial refund for unused seat licenses from previous quarter",
                "How do we export all our workspace data before closing our company account?",
                "Please ensure auto-renewal is turned off for our yearly subscription"
            ],
            "low": [
                "What is your cancellation policy if we decide to pause subscription next quarter?",
                "How many days before renewal do we need to submit a cancellation notice?",
                "If we cancel, do we still retain read-only access to historical dashboard reports?",
                "Can we pause our subscription for 2 months during our seasonal business hiatus?",
                "Where is the cancellation request button located inside billing settings?"
            ]
        }
    }
    
    # Customer text variation noise templates
    prefixes = [
        "Hello support team, ", "Hi there, ", "Urgent assistance needed: ", "Dear Customer Service, ",
        "Hey, ", "Help please! ", "Greetings, ", "Attention: ", "FYI, ", ""
    ]
    suffixes = [
        " Please look into this as soon as possible.", " Thanks for your prompt help.",
        " This is impacting our business operations.", " Let me know the resolution.",
        " Appreciate your assistance.", " Awaiting your quick reply.", " Regards.", ""
    ]
    
    records = []
    ticket_id = 10001
    
    categories = list(categories_data.keys())
    # Category distribution weights
    cat_weights = [0.28, 0.24, 0.20, 0.16, 0.12]
    
    for _ in range(n_samples):
        cat = np.random.choice(categories, p=cat_weights)
        
        # Priority distribution (Imbalance: High ~25%, Medium ~45%, Low ~30%)
        prio = np.random.choice(["high", "medium", "low"], p=[0.25, 0.45, 0.30])
        
        base_phrase = np.random.choice(categories_data[cat][prio])
        prefix = np.random.choice(prefixes)
        suffix = np.random.choice(suffixes)
        
        # Add random subtle perturbations (e.g. ticket numbers, random tokens)
        ticket_text = f"{prefix}{base_phrase}{suffix}"
        
        # Realistic metadata
        channel = np.random.choice(["Web Portal", "Email", "In-App Chat"], p=[0.5, 0.35, 0.15])
        customer_tier = np.random.choice(["Enterprise", "SMB", "Free Tier"], p=[0.3, 0.5, 0.2])
        
        records.append({
            "Ticket_ID": f"TKT-{ticket_id}",
            "Ticket_Text": ticket_text,
            "Category": cat,
            "Priority": prio.capitalize(),
            "Channel": channel,
            "Customer_Tier": customer_tier
        })
        ticket_id += 1
        
    df = pd.DataFrame(records)
    return df

def save_and_load_tickets(filepath: str) -> pd.DataFrame:
    """Ensures data directory exists and returns ticket dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        df = generate_support_ticket_dataset()
        df.to_csv(filepath, index=False)
        print(f"[DataLoader] Support ticket dataset generated and saved to {filepath} (Shape: {df.shape})")
    else:
        df = pd.read_csv(filepath)
        print(f"[DataLoader] Support ticket dataset loaded from {filepath} (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "data", "customer_support_tickets.csv")
    df = save_and_load_tickets(path)
    print(df.head())
