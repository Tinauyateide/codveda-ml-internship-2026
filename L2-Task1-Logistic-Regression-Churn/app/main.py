from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'churn_model.joblib'


class ChurnInput(BaseModel):
    state: str = Field(alias='State')
    account_length: int = Field(alias='Account length')
    area_code: int = Field(alias='Area code')
    international_plan: str = Field(alias='International plan')
    voice_mail_plan: str = Field(alias='Voice mail plan')
    number_vmail_messages: int = Field(alias='Number vmail messages')
    total_day_minutes: float = Field(alias='Total day minutes')
    total_day_calls: int = Field(alias='Total day calls')
    total_day_charge: float = Field(alias='Total day charge')
    total_eve_minutes: float = Field(alias='Total eve minutes')
    total_eve_calls: int = Field(alias='Total eve calls')
    total_eve_charge: float = Field(alias='Total eve charge')
    total_night_minutes: float = Field(alias='Total night minutes')
    total_night_calls: int = Field(alias='Total night calls')
    total_night_charge: float = Field(alias='Total night charge')
    total_intl_minutes: float = Field(alias='Total intl minutes')
    total_intl_calls: int = Field(alias='Total intl calls')
    total_intl_charge: float = Field(alias='Total intl charge')
    customer_service_calls: int = Field(alias='Customer service calls')

    model_config = {
        'populate_by_name': True,
        'json_schema_extra': {
            'example': {
                'State': 'KS',
                'Account length': 128,
                'Area code': 415,
                'International plan': 'No',
                'Voice mail plan': 'Yes',
                'Number vmail messages': 25,
                'Total day minutes': 265.1,
                'Total day calls': 110,
                'Total day charge': 45.07,
                'Total eve minutes': 197.4,
                'Total eve calls': 99,
                'Total eve charge': 16.78,
                'Total night minutes': 244.7,
                'Total night calls': 91,
                'Total night charge': 11.01,
                'Total intl minutes': 10.0,
                'Total intl calls': 3,
                'Total intl charge': 2.7,
                'Customer service calls': 1,
            }
        },
    }


app = FastAPI(title='Churn Prediction API', version='1.0.0')

model_bundle = joblib.load(MODEL_PATH)
pipeline = model_bundle['pipeline']
feature_columns = model_bundle['feature_columns']


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: ChurnInput) -> dict[str, float | bool]:
    row = payload.model_dump(by_alias=True)
    df = pd.DataFrame([row], columns=feature_columns)

    churn_probability = float(pipeline.predict_proba(df)[0, 1])
    will_churn = bool(churn_probability >= 0.5)

    return {
        'will_churn': will_churn,
        'churn_probability': round(churn_probability, 4),
    }