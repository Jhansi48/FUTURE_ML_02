# SupportSense NLP — Customer Support Triage & SLA Operations Report

## Executive Summary
This report details the implementation, empirical validation, and operational routing architecture of the **SupportSense NLP** intelligent ticket classification engine.
Operating dual classification heads for **Category Classification** and **Priority Tagging**, the system achieves:
- **Category Triage (Multinomial Naive Bayes):** Macro F1 of **1.0000** (Accuracy: **1.0000**).
- **Priority Tagging (Logistic Regression):** Macro F1 of **1.0000** (Accuracy: **1.0000**).

## Dataset Integrity & Sourcing
- **Dataset Origin**: Curated multi-domain enterprise customer support ticket dataset (3,500 total records across 5 categories and 3 priority tiers).
- **Deduplication Audit**: 0 exact duplicates and 0 cross-split duplicates verified.
- **Leakage Prevention**: TF-IDF feature space (5,000 max n-grams) is fitted strictly on the 80% training set. Target fields are isolated from input features.

## Category Model Comparison
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Multinomial Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Linear SVC (Calibrated) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM Classifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9886 | 0.9923 | 0.9856 | 0.9886 | 0.9885 |

## Priority Model Comparison
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Linear SVC (Calibrated) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM Classifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9857 | 0.9898 | 0.9820 | 0.9857 | 0.9857 |

## Operational Impact & Routing Logic
1. **Dynamic Queue Dispatch**: Maps joint predictions to specialized support queues with deterministic 1h to 24h SLA targets.
2. **Confidence-Gated Human Escalation**: Calibrated probabilities from Platt-scaled models trigger human review whenever joint confidence falls below 60%.
