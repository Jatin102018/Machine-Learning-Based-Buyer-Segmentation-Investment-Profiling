"""
Parcl Co. Limited — ML-Based Buyer Segmentation & Investment Profiling Dashboard
Run locally with:  streamlit run app.py
Requires: client_clustered.csv, cluster_profile.csv, cluster_eval.csv in the same folder.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Parcl | Buyer Segmentation", layout="wide", page_icon="🏢")

SEG_COLORS = {
    'Global Investors': '#5B3A8E',
    'Home-Focused / First-Time Buyers': '#2E86AB',
    'Corporate Buyers': '#E8862E',
    'High-Value / Luxury Investors': '#3C9D6B',
}


@st.cache_data
def load_data():
    data = pd.read_csv("client_clustered.csv")
    profile = pd.read_csv("cluster_profile.csv")
    cluster_eval = pd.read_csv("cluster_eval.csv")
    return data, profile, cluster_eval


data_all, profile, cluster_eval = load_data()

# ============================================================
# SIDEBAR — GLOBAL FILTERS
# ============================================================
st.sidebar.title("🏢 Parcl Co.")
st.sidebar.caption("Buyer Segmentation & Investment Profiling")
st.sidebar.markdown("---")

countries = sorted(data_all["country"].unique())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)

regions_available = sorted(data_all[data_all["country"].isin(sel_countries)]["region"].unique())
sel_regions = st.sidebar.multiselect("Region", regions_available, default=regions_available)

purposes = sorted(data_all["acquisition_purpose"].unique())
sel_purpose = st.sidebar.multiselect("Acquisition Purpose", purposes, default=purposes)

client_types = sorted(data_all["client_type"].unique())
sel_ctype = st.sidebar.multiselect("Client Type", client_types, default=client_types)

segments = list(SEG_COLORS.keys())
sel_segments = st.sidebar.multiselect("Buyer Segment", segments, default=segments)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Segments were discovered via K-Means clustering (k=4) on client demographic "
    "and purchase-behavior features, validated against Hierarchical clustering "
    "(Adjusted Rand Index ≈ 0.69). See Research Paper for full methodology."
)

data = data_all[
    (data_all["country"].isin(sel_countries)) &
    (data_all["region"].isin(sel_regions)) &
    (data_all["acquisition_purpose"].isin(sel_purpose)) &
    (data_all["client_type"].isin(sel_ctype)) &
    (data_all["Segment"].isin(sel_segments))
].copy()

if data.empty:
    st.warning("No clients match the current filters. Please widen your selection.")
    st.stop()

# ============================================================
# HEADER + KPI ROW
# ============================================================
st.title("Machine Learning–Based Buyer Segmentation & Investment Profiling")
st.caption(f"{len(data):,} clients matching current filters · {data_all.shape[0]:,} total clients analyzed")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients", f"{len(data):,}")
k2.metric("Total Investment", f"${data['Total_Investment_USD'].sum():,.0f}")
k3.metric("Avg Investment / Client", f"${data['Total_Investment_USD'].mean():,.0f}")
k4.metric("Avg Satisfaction", f"{data['satisfaction_score'].mean():.2f} / 5")
k5.metric("Segments Represented", f"{data['Segment'].nunique()}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview", "💰 Investor Behavior Dashboard",
    "🌍 Geographic Buyer Analysis", "🔍 Segment Insights Panel",
])

# ============================================================
# TAB 1 — BUYER SEGMENTATION OVERVIEW
# ============================================================
with tab1:
    st.subheader("Cluster Distribution")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        seg_counts = data["Segment"].value_counts().reindex(segments).dropna()
        fig = px.pie(values=seg_counts.values, names=seg_counts.index,
                     color=seg_counts.index, color_discrete_map=SEG_COLORS,
                     title="Client Distribution by Segment", hole=0.4)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.scatter(data, x="PCA1", y="PCA2", color="Segment",
                          color_discrete_map=SEG_COLORS, opacity=0.65,
                          title="Segments Visualized (PCA 2D Projection)",
                          hover_data=["client_id", "Total_Investment_USD"])
        fig.update_layout(height=420)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Optimal Cluster Selection — Elbow Method & Silhouette Score")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(cluster_eval, x="k", y="inertia", markers=True,
                       title="Elbow Method (Inertia vs. k)", color_discrete_sequence=["#5B3A8E"])
        fig.add_vline(x=4, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.line(cluster_eval, x="k", y="silhouette", markers=True,
                       title="Silhouette Score vs. k", color_discrete_sequence=["#E8862E"])
        fig.add_vline(x=4, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, width='stretch')
    st.caption("k=4 was selected for direct alignment with the four target buyer personas defined in the project brief, "
               "and is well-supported by both the elbow curve and silhouette score (k=3 and k=4 are the top two candidates).")

    st.subheader("Segment Profile Summary")
    display_profile = profile.copy()
    display_profile = display_profile.rename(columns={
        "Segment_Name": "Segment", "Num_Clients": "Clients", "Avg_Age": "Avg Age",
        "Pct_Company": "% Company", "Pct_Investment_Purpose": "% Investment Purpose",
        "Pct_Loan_Applied": "% Loan Applied", "Avg_Satisfaction": "Avg Satisfaction",
        "Avg_Num_Properties": "Avg # Properties", "Avg_Total_Investment": "Avg Total Investment ($)",
        "Avg_Purchase_Price": "Avg Purchase Price ($)", "Avg_Floor_Area": "Avg Floor Area (sqft)",
    })[["Segment", "Clients", "Avg Age", "% Company", "% Investment Purpose", "% Loan Applied",
        "Avg Satisfaction", "Avg # Properties", "Avg Total Investment ($)", "Avg Purchase Price ($)", "Avg Floor Area (sqft)"]]
    st.dataframe(
        display_profile.style.format({
            "Avg Age": "{:.1f}", "% Company": "{:.1f}", "% Investment Purpose": "{:.1f}",
            "% Loan Applied": "{:.1f}", "Avg Satisfaction": "{:.2f}", "Avg # Properties": "{:.2f}",
            "Avg Total Investment ($)": "{:,.0f}", "Avg Purchase Price ($)": "{:,.0f}", "Avg Floor Area (sqft)": "{:,.0f}",
        }),
        width='stretch',
    )

# ============================================================
# TAB 2 — INVESTOR BEHAVIOR DASHBOARD
# ============================================================
with tab2:
    st.subheader("Investment Patterns by Segment")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(data, x="Segment", y="Avg_Purchase_Price_USD", color="Segment",
                      color_discrete_map=SEG_COLORS, title="Purchase Price Distribution by Segment",
                      category_orders={"Segment": segments})
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.box(data, x="Segment", y="Total_Investment_USD", color="Segment",
                      color_discrete_map=SEG_COLORS, title="Total Investment Distribution by Segment",
                      category_orders={"Segment": segments})
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        loan_by_seg = data.groupby("Segment")["Loan_Applied_Flag"].mean().reindex(segments).dropna() * 100
        fig = px.bar(x=loan_by_seg.index, y=loan_by_seg.values, color=loan_by_seg.index,
                      color_discrete_map=SEG_COLORS, title="Loan Applied Rate by Segment (%)",
                      labels={"x": "", "y": "% Applied for Loan"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        cat_pref = pd.crosstab(data["Segment"], data["Preferred_Unit_Category"], normalize="index") * 100
        cat_pref = cat_pref.reindex(segments).dropna(how="all")
        fig = px.bar(cat_pref, barmode="stack", title="Preferred Unit Category by Segment (%)",
                      color_discrete_sequence=["#3C9D6B", "#E8862E"])
        st.plotly_chart(fig, width='stretch')

    st.subheader("Purchase Volume vs. Investment Size")
    fig = px.scatter(data, x="Num_Properties_Purchased", y="Total_Investment_USD", color="Segment",
                      color_discrete_map=SEG_COLORS, size="Avg_Floor_Area_Sqft", opacity=0.6,
                      hover_data=["client_id", "country"],
                      title="Number of Properties vs. Total Investment (bubble size = avg floor area)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Financing Behavior")
    finance_tab = pd.crosstab(data["Segment"], data["loan_applied"], normalize="index").round(3) * 100
    st.dataframe(finance_tab.reindex(segments).dropna(how="all").style.format("{:.1f}%"), width='stretch')

# ============================================================
# TAB 3 — GEOGRAPHIC BUYER ANALYSIS
# ============================================================
with tab3:
    st.subheader("Buyer Segments by Country")
    country_seg = pd.crosstab(data["country"], data["Segment"])
    country_seg_pct = country_seg.div(country_seg.sum(axis=1), axis=0) * 100

    metric_view = st.radio("View:", ["Client Count", "% Mix within Country"], horizontal=True)
    plot_df = country_seg if metric_view == "Client Count" else country_seg_pct
    fig = px.bar(plot_df, barmode="stack" if metric_view == "Client Count" else "stack",
                  color_discrete_map=SEG_COLORS, title=f"Segment Composition by Country ({metric_view})")
    fig.update_layout(height=480)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Investment Concentration by Country")
    country_inv = data.groupby("country", as_index=False).agg(
        Total_Investment=("Total_Investment_USD", "sum"),
        Avg_Investment=("Total_Investment_USD", "mean"),
        Clients=("client_id", "count"),
    ).sort_values("Total_Investment", ascending=False)
    fig = px.bar(country_inv, x="country", y="Total_Investment", color="Total_Investment",
                  color_continuous_scale="Purples", title="Total Investment by Country (USD)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Region Drill-Down")
    drill_country = st.selectbox("Select a country:", sorted(data["country"].unique()))
    region_df = data[data["country"] == drill_country]
    region_seg = region_df.groupby(["region", "Segment"]).size().unstack(fill_value=0)
    st.dataframe(region_seg, width='stretch')

    st.dataframe(
        country_inv.style.format({"Total_Investment": "${:,.0f}", "Avg_Investment": "${:,.0f}"}),
        width='stretch',
    )

# ============================================================
# TAB 4 — SEGMENT INSIGHTS PANEL
# ============================================================
with tab4:
    st.subheader("Descriptive Statistics per Segment")
    sel_seg_detail = st.selectbox("Choose a segment to inspect:", segments)
    seg_df = data[data["Segment"] == sel_seg_detail]

    if len(seg_df) == 0:
        st.info("No clients in this segment under current filters.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Clients", f"{len(seg_df):,}")
        d2.metric("Avg Age", f"{seg_df['Age'].mean():.1f}")
        d3.metric("Avg Total Investment", f"${seg_df['Total_Investment_USD'].mean():,.0f}")
        d4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f} / 5")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(seg_df, x="Age", nbins=20, title=f"Age Distribution — {sel_seg_detail}",
                                 color_discrete_sequence=[SEG_COLORS[sel_seg_detail]])
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(seg_df, x="satisfaction_score", nbins=5, title=f"Satisfaction Score — {sel_seg_detail}",
                                 color_discrete_sequence=[SEG_COLORS[sel_seg_detail]])
            st.plotly_chart(fig, width='stretch')

        st.subheader(f"Top Countries — {sel_seg_detail}")
        top_c = seg_df["country"].value_counts().head(8)
        fig = px.bar(x=top_c.index, y=top_c.values, color_discrete_sequence=[SEG_COLORS[sel_seg_detail]],
                      labels={"x": "Country", "y": "Clients"})
        st.plotly_chart(fig, width='stretch')

        st.subheader(f"Client Records — {sel_seg_detail}")
        st.dataframe(
            seg_df[["client_id", "client_type", "Age", "gender", "country", "region",
                    "acquisition_purpose", "loan_applied", "satisfaction_score",
                    "Num_Properties_Purchased", "Total_Investment_USD", "Preferred_Unit_Category"]]
            .sort_values("Total_Investment_USD", ascending=False),
            width='stretch', height=300,
        )

st.markdown("---")
st.caption("Parcl Co. Limited × Unified Mentor · ML-Based Buyer Segmentation & Investment Profiling · Built with Streamlit")
