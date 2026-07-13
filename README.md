# Sales Forecasting & Demand Intelligence System

## Project Overview

The Sales Forecasting & Demand Intelligence System is an end-to-end data analytics and machine learning project developed to analyze historical retail sales data, forecast future demand, detect sales anomalies, and segment products based on demand patterns. The project combines exploratory data analysis, time series forecasting, anomaly detection, clustering, and an interactive Streamlit dashboard to support data-driven business decisions.

## Streamlit Dashboard:
https://salesforecasting-xylofy-ai.streamlit.app/

---

## Objectives

- Analyze historical retail sales trends.
- Forecast future sales using multiple forecasting models.
- Detect unusual sales patterns using anomaly detection techniques.
- Segment products based on demand using clustering.
- Develop an interactive business intelligence dashboard using Streamlit.
- Generate business recommendations for inventory and supply chain management.

---

## Dataset

### Primary Dataset
- Retail Sales Dataset (`train.csv`)

### Supplementary Dataset
- Video Game Sales Dataset (`vgsales.csv`)

The supplementary dataset was used to validate the robustness of the anomaly detection pipeline by applying the same Isolation Forest and Z-Score techniques on an independent dataset.

---

## Project Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Time Series Analysis
6. Stationarity Testing (ADF Test)
7. Forecasting Models
   - SARIMA
   - Prophet
   - XGBoost
8. Model Evaluation
9. Anomaly Detection
   - Isolation Forest
   - Z-Score
10. Product Demand Segmentation
    - K-Means Clustering
    - PCA Visualization
11. Business Insights & Recommendations
12. Interactive Streamlit Dashboard

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

## Machine Learning Models

### Forecasting Models

- SARIMA
- Prophet
- XGBoost

### Anomaly Detection

- Isolation Forest
- Z-Score

### Clustering

- K-Means Clustering
- Principal Component Analysis (PCA)

---

## Model Performance

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 18,031.40 | 19,009.18 | 18.97% |
| Prophet | 20,250.79 | 22,318.41 | 21.86% |
| **XGBoost** | **15,169.05** | **19,040.85** | **14.79%** |

**Best Model:** XGBoost

---

## Three-Month Sales Forecast

| Forecast Month | Predicted Sales |
|---------------|----------------:|
| Month 1 | 85,865.91 |
| Month 2 | 86,567.86 |
| Month 3 | 88,749.48 |

---

## Key Findings

- Sales showed clear seasonal patterns across the dataset.
- Technology products generated the highest revenue.
- XGBoost achieved the best forecasting accuracy among all models.
- Anomaly detection successfully identified unusual sales spikes and drops.
- Product clustering classified products into High, Medium, and Low demand categories.
- Demand segmentation can improve inventory planning and warehouse utilization.

---

## Business Recommendations

- Use XGBoost for future sales forecasting.
- Maintain higher inventory for high-demand products.
- Reduce excess inventory for low-demand products.
- Monitor anomalies regularly to identify unexpected business events.
- Update forecasting models periodically using newly available sales data.

---

## Challenges Faced

- Handling date conversion and preprocessing for time series forecasting.
- Comparing multiple forecasting models and selecting the best-performing one.
- Implementing anomaly detection and interpreting unusual sales observations.
- Integrating the supplementary Video Game Sales dataset. Since both datasets had different structures and business contexts, direct merging was not feasible. Instead, the same anomaly detection pipeline was applied independently to validate the robustness of the approach.
- Developing the Streamlit dashboard and resolving issues related to deprecated APIs, data loading, and Git version control.

---

## Streamlit Dashboard Features

- Interactive KPI Dashboard
- Sales Overview
- Forecasting Dashboard
- Model Comparison
- Three-Month Forecast Visualization
- Anomaly Detection
- Product Demand Segmentation
- Business Recommendations
- Interactive Filters
- About Project Section

---

## Project Structure

```
Sales_Forecasting_Project/

│── app.py
│── requirements.txt
│── README.md
│── train.csv
│── vgsales.csv
│── comparison.csv
│── monthly_sales.csv
│── weekly_sales.csv
│── clusters.csv
│── styles.css
│── Sales_Forecasting_&_Demand_Intelligence_System.ipynb
│── summary.pdf
│
├── charts/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/lavanyasaxena01/Xylofy-AI-Project-3-4.git
```

Navigate to the project directory:

```bash
cd Xylofy-AI-Project-3-4
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## Future Improvements

- Integrate external factors such as holidays, promotions, and economic indicators.
- Deploy the forecasting model using cloud infrastructure.
- Automate model retraining with newly available data.
- Add real-time sales monitoring and alert notifications.
- Enhance dashboard customization with advanced filtering and reporting features.

---

## Author

**Lavanya Saxena**

B.Tech Artificial Intelligence & Data Science

Dr. Akhilesh Das Gupta Institute of Professional Studies

GitHub: https://github.com/lavanyasaxena01

LinkedIn: https://www.linkedin.com/in/lavanya-saxena-6b9b96240/

---

## Conclusion

This project demonstrates an end-to-end implementation of sales forecasting, anomaly detection, product demand segmentation, and business intelligence visualization. The developed system provides valuable insights that can assist retail organizations in improving inventory management, forecasting future demand, and making informed supply chain decisions.
