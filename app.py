import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecast & Demand Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Sales Forecast & Demand Intelligence Dashboard")

st.markdown(
"""
This dashboard presents the complete analysis of the **Sales Forecasting & Demand Intelligence System**.

The project includes:

- Exploratory Data Analysis
- Time Series Forecasting
- Anomaly Detection
- Product Demand Segmentation
"""
)

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_main():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df

@st.cache_data
def load_comparison():

    return pd.read_csv("comparison.csv")

@st.cache_data
def load_monthly():

    monthly = pd.read_csv("monthly_sales.csv")

    monthly["Order Date"] = pd.to_datetime(
        monthly["Order Date"]
    )

    return monthly

@st.cache_data
def load_weekly():

    weekly = pd.read_csv("weekly_sales.csv")

    weekly["Order Date"] = pd.to_datetime(
        weekly["Order Date"]
    )

    return weekly

@st.cache_data
def load_clusters():

    return pd.read_csv("clusters.csv")


df = load_main()

comparison = load_comparison()

monthly_sales = load_monthly()

weekly_sales = load_weekly()

clusters = load_clusters()

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "Home",

        "Sales Overview",

        "Forecasting",

        "Anomaly Detection",

        "Demand Segmentation",

        "About"

    ]

)
st.sidebar.markdown("---")

selected_region = st.sidebar.multiselect(

    "Region",

    options=df["Region"].unique(),

    default=df["Region"].unique()

)

selected_category = st.sidebar.multiselect(

    "Category",

    options=df["Category"].unique(),

    default=df["Category"].unique()

)

filtered = df[

    (df["Region"].isin(selected_region)) &

    (df["Category"].isin(selected_category))

]
if page=="Home":

    st.header("Dashboard Overview")

    col1,col2,col3,col4=st.columns(4)

    col1.metric(

        "Total Sales",

        f"${filtered['Sales'].sum():,.0f}"

    )

    col2.metric(

        "Total Orders",

        len(filtered)

    )

    col3.metric(

        "Categories",

        filtered["Category"].nunique()

    )

    shipping = (

        filtered["Ship Date"]

        if "Ship Date" in filtered.columns

        else None

    )

    col4.metric(

        "Regions",

        filtered["Region"].nunique()

    )

    st.markdown("---")

    st.subheader("Monthly Sales Trend")

    monthly = (

        filtered

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        monthly,

        x="Order Date",

        y="Sales",

        markers=True,

        title="Monthly Sales"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )
elif page=="Sales Overview":

    st.header("Sales Analysis")

    col1,col2=st.columns(2)

    category = (

        filtered

        .groupby("Category")["Sales"]

        .sum()

        .reset_index()

    )

    fig1 = px.bar(

        category,

        x="Category",

        y="Sales",

        color="Category",

        title="Revenue by Category"

    )

    col1.plotly_chart(

        fig1,

        width="stretch"

    )

    region = (

        filtered

        .groupby("Region")["Sales"]

        .sum()

        .reset_index()

    )

    fig2 = px.pie(

        region,

        values="Sales",

        names="Region",

        title="Region-wise Revenue"

    )

    col2.plotly_chart(

        fig2,

        width="stretch"

    )

    st.markdown("---")

    st.subheader("Top 10 Products")

    top = (

        filtered

        .groupby("Product Name")["Sales"]

        .sum()

        .sort_values(

            ascending=False

        )

        .head(10)

        .reset_index()

    )

    fig3 = px.bar(

        top,

        x="Sales",

        y="Product Name",

        orientation="h",

        title="Top Selling Products"

    )

    st.plotly_chart(

        fig3,

        width="stretch"

    )
# =====================================================
# FORECASTING PAGE
# =====================================================

elif page == "Forecasting":

    st.header("Sales Forecasting Dashboard")

    st.markdown(
    """
    This page compares the three forecasting models developed in this project.

    - SARIMA
    - Prophet
    - XGBoost

    The comparison is based on MAE, RMSE and MAPE.
    """
    )

    # -----------------------
    # Show Comparison Table
    # -----------------------

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        width="stretch"
    )

    # -----------------------
    # Best Model
    # -----------------------

    best_model = comparison.sort_values(
        "MAPE"
    ).iloc[0]

    st.success(
        f"""
        Best Forecasting Model

        **{best_model['Model']}**

        MAE : {best_model['MAE']:.2f}

        RMSE : {best_model['RMSE']:.2f}

        MAPE : {best_model['MAPE']:.2f}%
        """
    )

    st.markdown("---")

    # -----------------------
    # MAE Comparison
    # -----------------------

    st.subheader("Model Comparison by MAE")

    fig = px.bar(

        comparison,

        x="Model",

        y="MAE",

        color="Model",

        text_auto=True,

        title="Mean Absolute Error"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------
    # RMSE Comparison
    # -----------------------

    st.subheader("Model Comparison by RMSE")

    fig = px.bar(

        comparison,

        x="Model",

        y="RMSE",

        color="Model",

        text_auto=True,

        title="Root Mean Squared Error"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------
    # MAPE Comparison
    # -----------------------

    st.subheader("Model Comparison by MAPE")

    fig = px.bar(

        comparison,

        x="Model",

        y="MAPE",

        color="Model",

        text_auto=True,

        title="Mean Absolute Percentage Error"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.markdown("---")

    # -----------------------
    # Forecast Comparison
    # -----------------------

    st.subheader("3-Month Forecast Comparison")

    forecast = comparison[[
        "Model",
        "Forecast Month 1",
        "Forecast Month 2",
        "Forecast Month 3"
    ]]

    forecast_long = forecast.melt(

        id_vars="Model",

        var_name="Month",

        value_name="Forecast"

    )

    fig = px.line(

        forecast_long,

        x="Month",

        y="Forecast",

        color="Model",

        markers=True,

        title="3-Month Sales Forecast"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.markdown("---")

    # -----------------------
    # Business Interpretation
    # -----------------------

    st.subheader("Business Insights")

    st.info(
        """
• XGBoost achieved the lowest forecasting error and was selected as the final forecasting model.

• SARIMA successfully captured seasonality but produced slightly higher forecasting errors.

• Prophet effectively modeled trend and seasonality, although its accuracy was lower than XGBoost.

• Machine learning based forecasting proved more effective than traditional statistical forecasting for this dataset.

• The forecast can help businesses optimize inventory planning, improve demand estimation, and reduce stock shortages.
        """
    )
# =====================================================
# ANOMALY DETECTION
# =====================================================

elif page == "Anomaly Detection":

    st.header("Anomaly Detection")

    st.markdown("""
This page identifies unusual sales behaviour using the Isolation Forest algorithm.
Such anomalies may represent sudden demand spikes, promotional events, seasonal peaks,
or unexpected declines in sales.
""")

    # Weekly Sales Trend

    st.subheader("Weekly Sales Trend")

    fig = px.line(
        weekly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Weekly Sales"
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Isolation Forest Results

    if "Anomaly" in weekly_sales.columns:

        st.subheader("Detected Anomalies")

        fig = px.scatter(

            weekly_sales,

            x="Order Date",

            y="Sales",

            color="Anomaly",

            title="Isolation Forest Detection",

            color_discrete_map={
                "Normal":"blue",
                "Anomaly":"red"
            }

        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        anomaly_table = weekly_sales[
            weekly_sales["Anomaly"]=="Anomaly"
        ]

        st.subheader("Anomalous Weeks")

        st.dataframe(
            anomaly_table,
            width="stretch"
        )

    else:

        st.warning(
            "Anomaly column not found in weekly_sales.csv"
        )

    st.markdown("---")

    st.subheader("Business Insights")

    st.info("""

• Most weeks follow a consistent sales trend.

• The highlighted weeks represent unusually high or low sales.

• These anomalies may indicate promotions, festive seasons, stock shortages or unexpected demand.

• Businesses should investigate these periods to improve future forecasting and inventory planning.

""")

# =====================================================
# PRODUCT DEMAND SEGMENTATION
# =====================================================

elif page == "Demand Segmentation":

    st.header("Product Demand Segmentation")

    st.markdown("""

Products have been grouped using K-Means Clustering.

The segmentation helps identify

• High Demand Products

• Medium Demand Products

• Low Demand Products

""")

    st.subheader("Clustered Products")

    st.dataframe(
        clusters,
        width="stretch"
    )

    st.markdown("---")

    # Cluster Count

    if "Demand_Level" in clusters.columns:

        cluster_count = (

            clusters["Demand_Level"]

            .value_counts()

            .reset_index()

        )

        cluster_count.columns = [

            "Demand Level",

            "Count"

        ]

        fig = px.bar(

            cluster_count,

            x="Demand Level",

            y="Count",

            color="Demand Level",

            text_auto=True,

            title="Products in Each Demand Cluster"

        )

        st.plotly_chart(

            fig,

            width="stretch"

        )

    st.markdown("---")

    # Sales by Cluster

    if "Demand_Level" in clusters.columns and "Sales" in clusters.columns:

        fig = px.bar(

            clusters,

            x="Demand_Level",

            y="Sales",

            color="Demand_Level",

            title="Sales by Demand Level"

        )

        st.plotly_chart(

            fig,

            width="stretch"

        )

    st.markdown("---")

    st.subheader("Inventory Recommendations")

    st.success("""

High Demand Products

• Maintain higher inventory levels

• Prioritize stock replenishment

• Monitor demand frequently

Medium Demand Products

• Maintain balanced inventory

• Review sales trends monthly

Low Demand Products

• Reduce inventory holding costs

• Consider promotional offers

• Adopt Just-in-Time inventory management

""")
# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About":

    st.header("About the Project")

    st.markdown("""
## Sales Forecasting & Demand Intelligence System

This project was developed to analyze historical retail sales data and generate business insights using machine learning and time series forecasting techniques.

The dashboard helps businesses:

- Monitor sales performance
- Forecast future demand
- Detect unusual sales behaviour
- Segment products based on demand
- Support inventory and supply chain planning
""")

    st.markdown("---")

    st.subheader("Project Objectives")

    st.markdown("""
- Perform Exploratory Data Analysis (EDA)
- Analyze sales trends and seasonality
- Forecast future sales using multiple forecasting models
- Detect sales anomalies using machine learning
- Segment products into demand clusters
- Build an interactive business dashboard
""")

    st.markdown("---")

    st.subheader("Forecasting Models")

    models = pd.DataFrame({
        "Model": [
            "SARIMA",
            "Prophet",
            "XGBoost"
        ],
        "Purpose": [
            "Time Series Forecasting",
            "Trend & Seasonality Forecasting",
            "Machine Learning Forecasting"
        ]
    })

    st.dataframe(models, width="stretch")

    st.markdown("---")

    st.subheader("Machine Learning Techniques")

    techniques = pd.DataFrame({

        "Technique":[

            "Isolation Forest",

            "Z-Score",

            "K-Means",

            "PCA"

        ],

        "Application":[

            "Anomaly Detection",

            "Statistical Outlier Detection",

            "Demand Segmentation",

            "Cluster Visualization"

        ]

    })

    st.dataframe(
        techniques,
        width="stretch"
    )

    st.markdown("---")

    st.subheader("Technologies Used")

    st.markdown("""
- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-Learn
- Statsmodels
- Prophet
- XGBoost
- Matplotlib
""")

    st.markdown("---")

    st.subheader("Business Recommendations")

    st.success("""

**Sales Forecasting**

- Use XGBoost for future sales prediction.
- Update the forecasting model monthly.

**Inventory Management**

- Maintain higher inventory for high-demand products.
- Reduce inventory for low-demand products.

**Supply Chain**

- Monitor shipping delays regularly.
- Investigate anomalies immediately.

**Business Growth**

- Increase inventory before festive seasons.
- Improve stock planning using forecasted demand.

""")

    st.markdown("---")

    st.subheader("Project Summary")

    st.info("""

This dashboard demonstrates an end-to-end data science workflow, including data preprocessing, exploratory data analysis, forecasting, anomaly detection, product demand segmentation, and business intelligence visualization. It enables organizations to make data-driven decisions for inventory management, demand forecasting, and strategic planning.

""")

    st.markdown("---")

    st.caption(
        "Developed using Streamlit | Sales Forecasting & Demand Intelligence System"
    )

# =====================================================
# FOOTER
# =====================================================

st.sidebar.markdown("---")

st.sidebar.success(
    "Dashboard Loaded Successfully"
)

st.sidebar.info(
    "Developed using Streamlit"
)
