"""Data generation and preprocessing utilities for the Marketing Analysis XGBoost project.

This project is designed as a portfolio-ready machine learning repository. Since the
original dataset is unavailable, the repository generates a realistic synthetic
marketing campaign dataset and uses it to demonstrate a complete ML workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def generate_synthetic_marketing_data(
    n_samples: int = 2500,
    random_state: int = RANDOM_STATE,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic marketing campaign dataset.

    Target variable:
        conversion: 1 if a user/customer converts, 0 otherwise.

    The generated data intentionally contains realistic marketing drivers such as
    ad spend, impressions, click-through rate, channel, region, customer segment,
    discount level, and prior engagement. These variables are combined through a
    non-linear probability function so XGBoost is a suitable modelling choice.
    """
    rng = np.random.default_rng(random_state)

    channels = np.array(["Email", "Instagram", "Google Ads", "Referral", "Organic", "SMS"])
    regions = np.array(["North", "South", "East", "West", "Central"])
    segments = np.array(["New", "Returning", "High Value", "Dormant"])
    campaign_types = np.array(["Awareness", "Conversion", "Retargeting", "Seasonal"])

    channel = rng.choice(channels, size=n_samples, p=[0.18, 0.22, 0.24, 0.10, 0.18, 0.08])
    region = rng.choice(regions, size=n_samples)
    customer_segment = rng.choice(segments, size=n_samples, p=[0.35, 0.30, 0.20, 0.15])
    campaign_type = rng.choice(campaign_types, size=n_samples, p=[0.25, 0.35, 0.25, 0.15])

    ad_spend = rng.gamma(shape=5.0, scale=70.0, size=n_samples).round(2)
    impressions = rng.normal(loc=12000, scale=4500, size=n_samples).clip(800, 40000).round().astype(int)
    ctr = rng.beta(a=2.2, b=18, size=n_samples).clip(0.005, 0.35)
    clicks = np.maximum(1, (impressions * ctr).round().astype(int))
    average_order_value = rng.normal(loc=58, scale=20, size=n_samples).clip(10, 180).round(2)
    discount_percent = rng.choice([0, 5, 10, 15, 20, 25, 30], size=n_samples, p=[0.18, 0.13, 0.20, 0.18, 0.15, 0.10, 0.06])
    prior_purchases = rng.poisson(lam=1.4, size=n_samples).clip(0, 10)
    website_visits_30d = rng.poisson(lam=5.0, size=n_samples).clip(0, 40)
    email_open_rate = rng.beta(a=2.5, b=7, size=n_samples).clip(0.01, 0.95)
    days_since_last_purchase = rng.exponential(scale=45, size=n_samples).clip(1, 365).round().astype(int)

    channel_effect = pd.Series(channel).map(
        {"Email": 0.30, "Instagram": 0.10, "Google Ads": 0.18, "Referral": 0.45, "Organic": 0.25, "SMS": 0.16}
    ).to_numpy()
    segment_effect = pd.Series(customer_segment).map(
        {"New": -0.12, "Returning": 0.22, "High Value": 0.55, "Dormant": -0.40}
    ).to_numpy()
    campaign_effect = pd.Series(campaign_type).map(
        {"Awareness": -0.18, "Conversion": 0.32, "Retargeting": 0.48, "Seasonal": 0.12}
    ).to_numpy()

    # Non-linear conversion likelihood. This creates realistic relationships:
    # higher engagement, relevant campaign type and prior purchases increase conversion;
    # long inactivity and low click quality reduce conversion.
    logit = (
        -2.25
        + channel_effect
        + segment_effect
        + campaign_effect
        + 2.8 * ctr
        + 0.08 * np.log1p(clicks)
        + 0.10 * prior_purchases
        + 0.045 * website_visits_30d
        + 0.55 * email_open_rate
        + 0.018 * discount_percent
        - 0.006 * days_since_last_purchase
        - 0.0015 * average_order_value
        + rng.normal(0, 0.35, size=n_samples)
    )

    probability = 1 / (1 + np.exp(-logit))
    conversion = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "channel": channel,
            "region": region,
            "customer_segment": customer_segment,
            "campaign_type": campaign_type,
            "ad_spend": ad_spend,
            "impressions": impressions,
            "clicks": clicks,
            "click_through_rate": ctr.round(4),
            "average_order_value": average_order_value,
            "discount_percent": discount_percent,
            "prior_purchases": prior_purchases,
            "website_visits_30d": website_visits_30d,
            "email_open_rate": email_open_rate.round(4),
            "days_since_last_purchase": days_since_last_purchase,
            "conversion": conversion,
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transformer for categorical and numeric columns."""
    target = "conversion"
    categorical_features = df.drop(columns=[target]).select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = df.drop(columns=[target]).select_dtypes(exclude=["object", "category"]).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # compatibility for older scikit-learn versions
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "conversion",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed dataframe to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
