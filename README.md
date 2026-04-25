# Vintage Delinquency & Net Loss Rate Dashboard

**Author:** Denzel C. Seals | Senior Data Analyst  
**Domain:** Consumer Credit / BNPL Portfolio Analytics  
**Stack:** Python · SQL · Pandas · Plotly · Seaborn  

---

## Overview

This project builds an end-to-end vintage analysis framework for a synthetic consumer credit (BNPL / credit card) portfolio. It replicates the core analytical workflow used by credit risk and portfolio management teams to monitor cohort health, identify early stress signals, and project loss trajectories.

The analysis covers:
- **Vintage delinquency curves** — 30, 60, and 90+ days past due (DPD) by origination cohort
- **Roll rates** — transition probabilities between delinquency buckets month over month
- **Net loss rates** — charge-off timing, recovery assumptions, and loss as a % of original balance
- **Period-over-period comparison** — cohort benchmarking across origination vintages

---

## Business Context

Credit portfolios age differently depending on when accounts were originated. A vintage-level view isolates cohort behavior from portfolio mix effects — critical for understanding whether deterioration is structural (underwriting) or cyclical (macro). This framework answers questions like:

- Are newer vintages performing better or worse than prior cohorts at the same loan age?
- At what month does the 90+ DPD curve typically peak?
- What is the projected net loss rate for Q3 2023 originations at 18 months on book?

---

## Project Structure

```
vintage-delinquency-dashboard/
│
├── data/
│   ├── raw/                    # Synthetic dataset (generated, not committed)
│   └── processed/              # Cleaned, analysis-ready parquet files
│
├── src/
│   ├── data_generator.py       # Synthetic portfolio data generator
│   ├── feature_engineering.py  # DPD bucketing, roll rate calc, loss build
│   └── visualization.py        # Reusable Plotly/Seaborn charting functions
│
├── notebooks/
│   └── vintage_analysis.ipynb  # Full analysis narrative (run top to bottom)
│
├── outputs/
│   └── charts/                 # Exported chart PNGs for README / portfolio
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/yourhandle/vintage-delinquency-dashboard.git
cd vintage-delinquency-dashboard
pip install -r requirements.txt

# 2. Generate synthetic data
python src/data_generator.py

# 3. Open and run the notebook
jupyter notebook notebooks/vintage_analysis.ipynb
```

---

## Key Outputs

| Chart | Description |
|---|---|
| Vintage DPD Curves | 30/60/90+ DPD rate by months on book, faceted by origination cohort |
| Roll Rate Heatmap | Month-over-month transition matrix across delinquency buckets |
| Net Loss Rate by Vintage | Cumulative net loss as % of original balance over loan life |
| Cohort Comparison | Overlaid vintage curves benchmarked against portfolio average |

---

## Methodology Notes

- **Synthetic data** is parameterized by cohort vintage, risk tier, origination balance, and macro stress factor — all configurable in `data_generator.py`
- **Charge-off assumption**: accounts at 120+ DPD are charged off in the following month
- **Recovery rate**: 15% applied 6 months post charge-off (configurable)
- **Loss rate denominator**: original outstanding balance at origination (consistent vintage methodology)

---

## Skills Demonstrated

- Cohort construction and vintage segmentation from transaction-level data
- DPD bucket engineering and delinquency state tracking
- Roll rate and transition matrix computation
- Net loss modeling with charge-off and recovery timing
- Time-series visualization of multi-cohort portfolio behavior
- SQL-equivalent logic implemented in Pandas for auditability
