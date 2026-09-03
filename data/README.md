# SupportSense NLP — Customer Support Ticket Dataset Specification

## Overview
This directory contains the customer support ticket dataset utilized by **SupportSense NLP** for dual-head classification (Ticket Category and Operational Priority Tagging).

## Dataset Schema
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Ticket_ID` | String | Unique identifier for each support ticket (`TKT-10001` to `TKT-13500`). |
| `Ticket_Text` | String | Raw customer ticket content including issue description, urgency signals, and context. |
| `Category` | Categorical | Target classification label (5 classes: `Technical Issues`, `Billing & Payments`, `Account Access`, `Product Inquiries`, `Cancellation & Refunds`). |
| `Priority` | Categorical | Operational priority tier (3 classes: `High`, `Medium`, `Low`). |
| `Channel` | Categorical | Originating support channel (`Web Portal`, `Email`, `In-App Chat`). |
| `Customer_Tier` | Categorical | Customer account tier (`Enterprise`, `SMB`, `Free Tier`). |

## Provenance & Integrity
- **Total Records:** 3,500 samples
- **Duplicates:** 0 exact duplicates and 0 cross-split duplicates
- **Isolation:** Target fields (`Category`, `Priority`) are strictly isolated from feature inputs.
