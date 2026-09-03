# Dataset Documentation: Customer Support Ticket Dataset

## Source & Description
This dataset captures 3,500 multi-channel customer support tickets representing real-world inquiries across SaaS, E-commerce, and FinTech domains.

## Attributes:
- `Ticket_ID`: Unique ticket identifier (`TKT-10001` to `TKT-13500`)
- `Ticket_Text`: Raw unstructured customer submission
- `Category`: Target 1 (5 classes: `Technical Issues`, `Billing & Payments`, `Account Access`, `Product Inquiries`, `Cancellation & Refunds`)
- `Priority`: Target 2 (3 classes: `High`, `Medium`, `Low`)
- `Channel`: Submission channel (`Web Portal`, `Email`, `In-App Chat`)
- `Customer_Tier`: Organization plan (`Enterprise`, `SMB`, `Free Tier`)

## Preprocessing Guidelines:
1. Normalize text and expand English contractions.
2. Remove punctuation and generic conversation stop words while preserving technical error tokens (`500`, `503`, `mfa`, `sso`, `sql`).
3. Apply stratified splitting to preserve joint class distributions.
