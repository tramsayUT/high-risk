# SDoH Extraction from Clinical Notes

High-risk project for AI 395T (AI for Healthcare, Fall 2026).
Compares regex, Bio_ClinicalBERT, and a local LLM for extracting social and
behavioral determinants of health from MIMIC-III discharge summaries, plus a
counterfactual fairness test.

## Data access
Requires PhysioNet credentialed access to MIMIC-III-Ext-SBDH v1.0.0 and
MIMIC-III (BigQuery). No data is included in this repo.

## Run order
1. notebooks/01_data_prep.ipynb