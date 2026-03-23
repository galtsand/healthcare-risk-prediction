🔷 Healthcare Risk Prediction
Overview

This project develops a healthcare risk stratification model to identify members at high risk of becoming high-cost within a future 6-month period. Using synthetic but realistic claims-based data, the solution combines longitudinal cost, utilization, and risk signals to generate actionable predictions.

The model achieves ~0.89 ROC-AUC and concentrates ~70% of high-cost members within the top decile, enabling highly targeted intervention strategies.

Business Problem

Healthcare spend is highly skewed, with a small percentage of members driving the majority of costs. Early identification of these members allows organizations to:

Target care management programs
Reduce avoidable utilization (ED visits, readmissions)
Improve outcomes under value-based care contracts
Optimize provider engagement and resource allocation
Data Model (Synthetic)

The dataset is designed to reflect real-world payer/provider systems.

Core Tables:

vbc_members – demographics, enrollment, risk indicators
vbc_member_month – longitudinal cost and RAF signals
vbc_medical_claims – claims, utilization, and clinical proxies
vbc_attribution_input – provider attribution and visit behavior
Feature Engineering

Features are dynamically generated using rolling historical windows (6–12 months):

Cost & Utilization

Total cost (6m / 12m)
Medical vs Rx spend
Utilization intensity (claims, visits, admits)
PMPM and cost aggregation metrics

Clinical & Risk

RAF scores
Diagnosis / medication complexity proxies
Readmission and acute event indicators

Derived Signals

Utilization intensity (aggregated activity signal)
Complexity score (risk + medication diversity)
Acute event flag (ED / inpatient activity)
Interaction features (cost × utilization)
Target Definition

High-Cost Next 6 Months Flag

Members are labeled as high-cost if their future 6-month cost exceeds a threshold derived from the cost distribution.

A learnable target construction approach ensures realistic class balance (~14–15%).

Modeling Approach

Models evaluated:

Logistic Regression (baseline)
Random Forest
XGBoost (primary model)

Performance:

ROC-AUC: ~0.89
Strong recall for high-cost members (~92%)
Effective precision in top risk tier
Model Performance (Key Insight)

Top 10% Risk Segment:

~70% of high-cost members captured
High precision targeting for intervention

Risk Decile Behavior:

Bottom deciles: near-zero high-cost rate
Top decile: ~70% high-cost rate

👉 Demonstrates strong ranking power and cost concentration

Explainability

SHAP (SHapley Additive Explanations) is used to interpret model predictions:

Global feature importance (top drivers of risk)
Feature impact direction (what increases/decreases risk)
Member-level explainability (why a specific member is flagged)

Key Drivers Identified:

Total cost (6m / 12m)
Utilization intensity
Inpatient and high-cost claim activity
RAF / risk indicators
Outputs
Member-level risk scores
Risk stratification (deciles)
Top high-risk cohort for intervention
Feature importance and SHAP explainability plots
Repository Structure
data/        # raw and processed datasets
sql/         # feature and target creation logic
notebooks/   # model development and analysis
outputs/     # predictions, SHAP plots, metrics
src/         # modular pipeline code
Future Enhancements
Incorporate pharmacy-specific signals (specialty Rx)
Add encounter-level features (DRG, LOS)
Enhance temporal modeling (sequence-based features)
Build interactive dashboard (Power BI / Streamlit)
Key Takeaway

This project demonstrates how healthcare data can be transformed into a predictive risk framework that:

Identifies high-cost members early
Enables targeted intervention
Improves cost efficiency
Supports value-based care strategies
