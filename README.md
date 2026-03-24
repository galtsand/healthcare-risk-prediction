Healthcare Risk Prediction
Overview

This project builds a predictive model to identify healthcare members at risk of becoming high-cost in a future period. Using synthetic but realistic healthcare data, the solution combines member demographics, longitudinal cost patterns, medical claims, and provider utilization behavior to generate actionable risk insights.

The goal is to support early intervention, care management, and cost containment strategies in both fee-for-service and value-based care environments.

Business Problem

Healthcare costs are highly concentrated, with a small percentage of members driving a large portion of total spend. Identifying these members early allows organizations to:

Target care management programs

Reduce avoidable utilization (e.g., ED visits, readmissions)

Improve financial performance under value-based contracts

Optimize provider engagement and care coordination

This project focuses on predicting future high-cost members using historical utilization and risk signals.

Data Model (Synthetic)

The project uses a synthetic healthcare dataset modeled after real-world payer/provider systems.

Core Tables

vbc_members – Member demographics, enrollment, and risk indicators

vbc_member_month – Longitudinal monthly cost and RAF signals

vbc_medical_claims – Detailed claims with utilization and clinical flags

vbc_attribution_input – Provider attribution and visit patterns

Feature Engineering

Features are built using rolling historical windows (e.g., last 6–12 months):

Cost & Utilization

Total medical and Rx cost

PMPM (per member per month)

Claim counts and cost trends

Clinical & Risk

RAF score (risk adjustment)

Chronic condition proxies (via diagnosis counts)

Readmissions and avoidable ED events

Provider Behavior

PCP vs specialist visit patterns

Professional cost signals

Attribution indicators

Target Definition

The model predicts:

High-Cost Next 6 Months Flag

A member is labeled as high-cost if their total allowed cost in a future 6-month window exceeds a defined threshold or falls within the top cost percentile.

Modeling Approach

## Results
- ROC-AUC: ~0.89
- Top decile captures ~70% of high-cost members
- Model effectively segments members into actionable risk tiers

The model demonstrates strong ability to identify high-cost members early, enabling targeted intervention strategies.

## Model Explainability (SHAP)

To support transparency and real-world healthcare decision-making, SHAP (SHapley Additive Explanations) was used to interpret model predictions.

Explainability is critical in healthcare to ensure models are auditable, trustworthy, and aligned with clinical and operational workflows.

### Key Insights

- Historical cost and utilization are the strongest predictors of future high-cost status
- Members with rising utilization trends show significantly higher risk
- Pharmacy and specialist utilization contribute meaningfully to risk stratification

### SHAP Summary Plot

![SHAP Summary](images/shap_summary.png)

This plot shows the global impact of features across all predictions.

### Feature Importance

![Feature Importance]([images/shap_bar.png](https://github.com/galtsand/healthcare-risk-prediction/blob/main/shap_summary.png))

Top variables contributing to model predictions.



Logistic Regression (baseline, interpretable)

Tree-based model (e.g., Random Forest or XGBoost)

Evaluation includes:

ROC-AUC

Precision / Recall

Cost concentration by risk tier

Outputs

Member-level risk scores

Risk stratification (Low / Medium / High)

Feature importance insights

Summary of cost distribution by risk segment

Repository Structure
data/        # raw and processed datasets
sql/         # feature and target dataset creation
notebooks/   # model development and analysis
outputs/     # scored results and metrics
src/         # optional modular Python code
Future Enhancements

Incorporate pharmacy claims (specialty Rx signals)

Add encounter-level data (DRG, LOS, service line)

Introduce model explainability (SHAP)

Build interactive dashboard (Power BI / Streamlit)

Related Work

This project complements additional healthcare analytics work focused on:

Value-Based Care (VBC) financial modeling

Fee-for-Service (FFS) encounter analytics

Data engineering pipelines for healthcare data platforms

Key Takeaway

This project demonstrates how healthcare data can be transformed into a predictive framework that supports proactive decision-making, cost control, and improved patient outcomes.
