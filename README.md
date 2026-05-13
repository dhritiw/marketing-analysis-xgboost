# Marketing Analysis using XGBoost

## Project Overview

This repository demonstrates an end-to-end machine learning workflow for marketing analytics using **XGBoost**. The project analyses marketing campaign data to predict whether a customer is likely to convert based on campaign, customer, engagement, and spend-related features.

Since the original project files are unavailable, this version includes a realistic synthetic marketing dataset generator and a complete working ML pipeline. The structure is designed to be portfolio-ready and easy for recruiters or hiring managers to review.

## Business Objective

Marketing teams often need to understand which campaigns, channels, and customer segments are most likely to drive conversion. This project uses machine learning to:

- Predict customer conversion likelihood
- Identify the strongest drivers of conversion
- Compare model performance before and after hyperparameter tuning
- Support better marketing budget allocation and campaign targeting decisions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- GridSearchCV

## Repository Structure

```text
marketing-analysis-xgboost/
│
├── data/
│   ├── raw/
│   │   └── synthetic_marketing_campaigns.csv
│   └── processed/
│       └── marketing_campaigns_processed.csv
│
├── notebooks/
│   └── marketing_analysis_xgboost.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   ├── model_metrics.json
│   └── xgboost_marketing_model.joblib
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Dataset

The dataset is synthetically generated to represent a realistic marketing campaign environment. It includes features such as:

- Marketing channel
- Region
- Customer segment
- Campaign type
- Ad spend
- Impressions
- Clicks
- Click-through rate
- Average order value
- Discount percentage
- Prior purchases
- Website visits in the last 30 days
- Email open rate
- Days since last purchase
- Conversion outcome

The target variable is:

```text
conversion
```

Where:

- `1` = customer converted
- `0` = customer did not convert

## Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Generate synthetic marketing campaign data
2. Explore and prepare the dataset
3. Split data into training and testing sets
4. Apply preprocessing for categorical and numeric variables
5. Train an XGBoost classifier
6. Tune hyperparameters using GridSearchCV
7. Evaluate the final model using classification metrics
8. Export visual outputs such as ROC curve, confusion matrix, and feature importance

## Model

The project uses **XGBoost Classifier**, a gradient boosting algorithm suitable for tabular data and non-linear relationships.

The model is tuned using **GridSearchCV** across parameters such as:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Feature Importance

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/marketing-analysis-xgboost.git
cd marketing-analysis-xgboost
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full training pipeline

```bash
python src/model_training.py
```

This will generate:

- Synthetic dataset
- Processed dataset
- Trained XGBoost model
- Model metrics JSON file
- Confusion matrix chart
- ROC curve chart
- Feature importance chart

### 5. Open the notebook

```bash
jupyter notebook notebooks/marketing_analysis_xgboost.ipynb
```

## Key Outputs

After running the project, outputs are saved in the `outputs/` folder:

- `model_metrics.json` contains final model scores and best hyperparameters
- `confusion_matrix.png` shows classification performance
- `roc_curve.png` shows the model's ability to separate converters from non-converters
- `feature_importance.png` shows the most influential marketing variables

## Example Business Interpretation

The model can help a marketing team understand which customer and campaign variables are most associated with conversion. For example, features such as prior purchases, recent website visits, email open rate, campaign type, and channel may strongly influence conversion likelihood.

These insights can support:

- Better campaign targeting
- Smarter budget allocation
- Improved customer segmentation
- Higher ROI on marketing spend
- More focused retargeting strategies

## Future Improvements

Potential improvements include:

- Adding SHAP values for explainable AI
- Comparing XGBoost with Random Forest, Logistic Regression, and LightGBM
- Building a Streamlit dashboard for marketing insights
- Adding customer lifetime value prediction
- Connecting the pipeline to a real marketing dataset
- Deploying the model as an API

## Author

Dhriti  
MSc Management, Imperial College London  
BE Information Science and Engineering  
Founder, Wooziee
