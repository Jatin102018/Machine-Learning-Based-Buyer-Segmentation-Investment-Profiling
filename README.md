# 🏢 ML-Based Buyer Segmentation & Investment Profiling for Real Estate

A machine learning project that discovers hidden buyer segments in real estate transaction data using K-Means and Hierarchical clustering, with an interactive Streamlit dashboard for live analytics.

## 📌 Project Overview
Real estate companies deal with highly diverse buyers — individual home buyers, institutional investors, corporate buyers, and luxury investors — but often lack a data-driven way to tell them apart. This project applies unsupervised ML to 2,000 clients and 10,000 property transactions to uncover natural buyer segments and profile their investment behavior.

## 🎯 Key Features
- **Data Cleaning**: Handled mixed date formats, currency-formatted fields, and merged transactional + demographic data
- **Feature Engineering**: Aggregated 7,305 sold transactions into client-level purchase behavior (total investment, avg price, floor area, etc.)
- **Clustering**: K-Means (primary) + Hierarchical clustering (validation) — cross-validated with Adjusted Rand Index (0.69)
- **Optimal k Selection**: Elbow Method + Silhouette Score analysis
- **Interactive Dashboard**: Streamlit app with 4 modules — Segmentation Overview, Investor Behavior, Geographic Analysis, Segment Insights

## 🧩 Discovered Segments
| Segment | Clients | Key Trait |
|---|---|---|
| Global Investors | 555 | 100% investment-purpose individuals |
| Home-Focused / First-Time Buyers | 779 | Largest segment, smallest ticket size |
| Corporate Buyers | 103 | Companies, most properties per client |
| High-Value / Luxury Investors | 563 | Highest investment size & floor area |

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit + Plotly (interactive dashboard)
- K-Means & Agglomerative (Hierarchical) Clustering

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Repository Structure
