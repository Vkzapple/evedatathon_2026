# DATATHON 2026 HS Track
## Competition Guide

> Competition Period
13 July 2026 - 2 August 2026

---

# Competition Overview

This competition consists of **2 independent tasks**.

Each task has:
- Separate Kaggle competition page
- Separate notebook
- Separate report
- Separate submission

⚠ Both tasks MUST be completed.

Only completing one task will significantly reduce the final preliminary score.

---

# Finalist Requirement

To maximize the preliminary score:

✅ Submit Task 1 prediction
✅ Submit Task 2 prediction
✅ Create Report Task 1
✅ Create Report Task 2

---

# Workflow

For EACH task:

Dataset
    ↓
EDA
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Validation
    ↓
Prediction
    ↓
submission.csv
    ↓
Submit to Kaggle
    ↓
Improve Model
    ↓
Repeat

---

# Kaggle Leaderboard

Leaderboard score uses approximately **50% of the test set**.

Final ranking uses the remaining hidden test data.

Meaning:

Public Leaderboard
≠
Final Score

Avoid overfitting to the public leaderboard.

---

# Submission

Each submission consists of:

submission.csv

NOT:
- notebook
- source code
- report

Leaderboard score is calculated automatically after submission.

---

# Entries

Entries = Number of submissions made.

Example:

Submission 1
Score 0.82

Submission 2
Score 0.85

Submission 3
Score 0.89

Entries = 3

---

# Submission Limit

Maximum:

3 submissions/day

Plan every submission carefully.

Recommended:

Submission 1
Baseline

Submission 2
Improved Features

Submission 3
Hyperparameter Tuning

Don't waste submissions.

---

# Notebook Runtime

Notebook runtime limit:

4 Hours

This DOES NOT mean competition duration.

Competition Duration:

13 July
↓

2 August

Notebook runtime only limits one execution.

---

# Notebook Requirement

All computations must run inside Kaggle Notebook.

Required:

- preprocessing
- feature engineering
- training
- validation
- inference
- submission generation

Everything should be reproducible.

---

# External Data Rule

Allowed:

✔ Competition dataset
✔ Public pretrained model
✔ Public pretrained weights

Not Allowed:

✘ External datasets
✘ Additional training data
✘ Self-collected data

---

# Pretrained Model Rule

Allowed if:

✔ Open-source
✔ Uploaded before 13 June 2026
✔ Inference runs locally inside notebook
✔ No external API

Examples:

- BERT
- YAMNet
- ResNet
- EfficientNet

---

# Beginner Notebook

Allowed:

- Read beginner notebook
- Learn from public notebook
- Copy baseline
- Improve the solution

Improvement is expected.

---

# Report

Task 1
↓

1 Report

Task 2
↓

1 Report

Maximum:

5 pages (content only)

Cover page and bibliography are excluded.

Report submission:
TBA

---

# Suggested Development Workflow

VS Code
↓

Coding

↓

Debugging

↓

Experiment

↓

Copy to Kaggle Notebook

↓

Run GPU

↓

Generate submission.csv

↓

Submit

---

# Daily Strategy

1. Read competition discussion.
2. Check leaderboard.
3. Improve feature engineering.
4. Compare new score.
5. Save notebook version.
6. Document experiment.
7. Submit only if improvement is meaningful.

---

# Experiment Log

| Version | Model | CV Score | LB Score | Notes |
|----------|-------|----------|----------|------|
| v1 | Baseline | | | |
| v2 | Random Forest | | | |
| v3 | XGBoost | | | |
| v4 | LightGBM | | | |
| v5 | Ensemble | | | |

---

# Things To Try

## Data Cleaning

- Missing Value
- Duplicate
- Outlier
- Encoding
- Scaling

---

## Feature Engineering

- New Features
- Feature Selection
- Target Encoding
- Interaction Feature

---

## Models

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Logistic Regression
- Neural Network (if suitable)

---

## Validation

- Train Validation Split
- Cross Validation
- Stratified KFold
- GroupKFold (if applicable)

---

## Hyperparameter Tuning

- Grid Search
- Random Search
- Optuna

---

# Before Every Submission

☐ Notebook runs from top to bottom

☐ No error

☐ submission.csv format correct

☐ Score validated

☐ Model saved

☐ Notebook version saved

☐ Experiment logged

---

# Before Competition Ends

☐ Task 1 submitted

☐ Task 2 submitted

☐ Report Task 1 completed

☐ Report Task 2 completed

☐ Final notebook saved

☐ Backup project

☐ Double-check leaderboard

---

Good luck 🚀