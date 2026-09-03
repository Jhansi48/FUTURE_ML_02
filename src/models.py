"""
SupportSense NLP - Model Architectures & Dual-Head Pipeline
Provides model definitions for Category Classification and Priority Tagging
with probability calibration and class balancing.
"""

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def get_category_models() -> Dict[str, Any]:
    """Returns candidate multi-class classification models for Ticket Category."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=1.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Linear SVC (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual=False),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    }

def get_priority_models() -> Dict[str, Any]:
    """Returns candidate classification models for Ticket Priority (High, Medium, Low)."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.2,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Linear SVC (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual=False),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    }

def stratified_train_test_split(
    df,
    test_size: float = 0.20,
    random_state: int = 42
):
    """Performs stratified train/test split preserving joint distribution."""
    # Create combined stratum for multi-head consistency
    df = df.copy()
    df["stratum"] = df["Category"].astype(str) + "_" + df["Priority"].astype(str)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["stratum"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
