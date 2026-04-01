from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'churn-bigml-80.csv'
TEST_PATH = ROOT / 'churn-bigml-20.csv'
MODEL_PATH = ROOT / 'models' / 'churn_model.joblib'

TARGET_COLUMN = 'Churn'
CATEGORICAL_COLUMNS = ['State', 'International plan', 'Voice mail plan']


def build_pipeline(numeric_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS),
            ('num', 'passthrough', numeric_columns),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=1,
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]
    numeric_columns = [
        col for col in feature_columns if col not in CATEGORICAL_COLUMNS
    ]

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(bool)
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(bool)

    pipeline = build_pipeline(numeric_columns)
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    probs = pipeline.predict_proba(x_test)[:, 1]

    print('Evaluation on test split (churn-bigml-20.csv):')
    print(classification_report(y_test, preds, digits=4))
    print(f'ROC AUC: {roc_auc_score(y_test, probs):.4f}')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        'pipeline': pipeline,
        'feature_columns': feature_columns,
    }
    joblib.dump(bundle, MODEL_PATH)
    print('Model saved to models/churn_model.joblib')


if __name__ == '__main__':
    main()
