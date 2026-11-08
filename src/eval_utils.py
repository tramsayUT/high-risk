from sklearn.metrics import f1_score, classification_report
import pandas as pd

LABEL_COLS: list[str] = [
    "sdoh_community_present", "sdoh_community_absent", "sdoh_education",
    "sdoh_economics", "sdoh_environment",
    "behavior_alcohol", "behavior_tobacco", "behavior_drug",
]

def score_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in LABEL_COLS:
        rows.append({
            "model": model_name,
            "label": col,
            "macro_f1": f1_score(y_true[col], y_pred[col], average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true[col], y_pred[col], average="weighted", zero_division=0),
            "n": len(y_true),
        })
    return pd.DataFrame(rows)

def per_class_report(y_true: pd.Series, y_pred: pd.Series) -> str:
    return classification_report(y_true, y_pred, zero_division=0)