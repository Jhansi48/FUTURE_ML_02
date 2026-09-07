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

def stratified_train_val_test_split(
    df: Any,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Performs stratified 3-way train/validation/test split (70/15/15) preserving joint distribution
    across Category and Priority combinations.
    """
    df = df.copy()
    df["stratum"] = df["Category"].astype(str) + "_" + df["Priority"].astype(str)
    
    # Split train vs temp (validation + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=df["stratum"]
    )
    
    # Split temp into validation and test (50/50 of the 30% remainder -> 15% each)
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_in_temp,
        random_state=random_state,
        stratify=temp_df["stratum"]
    )
    
    train_df = train_df.drop(columns=["stratum"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["stratum"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["stratum"]).reset_index(drop=True)
    
    return train_df, val_df, test_df

def stratified_train_test_split(
    df,
    test_size: float = 0.20,
    random_state: int = 42
):
    """Legacy 2-way split function for backward compatibility."""
    df = df.copy()
    df["stratum"] = df["Category"].astype(str) + "_" + df["Priority"].astype(str)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["stratum"]
    )
    return train_df.drop(columns=["stratum"]).reset_index(drop=True), test_df.drop(columns=["stratum"]).reset_index(drop=True)
