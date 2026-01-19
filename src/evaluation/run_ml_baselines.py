
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from config.config import load_config, PROJECT_ROOT
from src.data.iemocap_text_dataset import build_iemocap_text_dataframe
from src.utils.logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)

def get_data_splits(cfg):
    """
    Reconstruct the train/val/test splits used for the DL model
    to ensure fair comparison.
    """
    root_dir = PROJECT_ROOT / cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"

    logger.info("Building IEMOCAP text DataFrame...")
    df = build_iemocap_text_dataframe(root_dir)

    # Drop rows with NaN
    df = df.dropna(subset=["Emotion", "Utterance"])

    # Split into train / temp (80/20)
    train_df, temp_df = train_test_split(
        df,
        train_size=0.8,
        test_size=0.2,
        stratify=df["Emotion"],
        random_state=cfg.seed,
    )

    # Split temp into val / test (50/50 of temp = 10/10 of total)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        test_size=0.5,
        stratify=temp_df["Emotion"],
        random_state=cfg.seed,
    )

    logger.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def train_and_evaluate():
    cfg = load_config()
    train_df, val_df, test_df = get_data_splits(cfg)

    # Merge train and val for traditional ML (more data is better, we tune less here)
    # OR keep them separate if we wanted to tune hyperparams.
    # For a quick baseline, let's just train on TRAIN and evaluate on TEST
    # to match the DL exact setup (trained on TRAIN).

    # Feature Extraction (TF-IDF)
    logger.info("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df["Utterance"])
    X_test = vectorizer.transform(test_df["Utterance"])

    y_train = train_df["Emotion"]
    y_test = test_df["Emotion"]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Support Vector Machine": SVC(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=cfg.seed),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=cfg.seed),
        "Multinomial NB": MultinomialNB(),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(class_names),
            random_state=cfg.seed,
            eval_metric='mlogloss'
        )
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train_enc)
            y_pred_enc = model.predict(X_test)

            acc = accuracy_score(y_test_enc, y_pred_enc)
            f1 = f1_score(y_test_enc, y_pred_enc, average='weighted')
            macro_f1 = f1_score(y_test_enc, y_pred_enc, average='macro')

            logger.info(f"{name} - Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

            results[name] = {
                "accuracy": acc,
                "weighted_f1": f1,
                "macro_f1": macro_f1,
                "report": classification_report(
                    y_test_enc,
                    y_pred_enc,
                    labels=range(len(class_names)),
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )
            }
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")

    # Save results
    output_dir = PROJECT_ROOT / "reports/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ml_baselines_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline results saved to {output_path}")
    return output_path

if __name__ == "__main__":
    train_and_evaluate()
