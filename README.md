# Subset Sum Finder (Streamlit)

A simple web app to find one or more subsets of numbers whose sum equals a target.

- Input via CSV/Excel (first column) or comma-separated text
- Choose how many distinct combinations to return
- Exact solution using Integer Programming
  - Tries PuLP (CBC) first
  - Falls back to OR-Tools CP-SAT if needed
- Download results as CSV or Excel

## Run locally
pip install -r requirements.txt
streamlit run subset_app.py

## Deployed
Deployed on Streamlit Community Cloud.
