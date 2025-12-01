# Aviation Data Analytics Project

A comprehensive data analytics project showcasing flight performance analysis, route optimization insights, and interactive visualizations. Perfect for demonstrating data analytics skills for the **Cathay Pacific Digital & IT Summer Internship Programme**.

## 🎯 Project Overview

This project analyzes flight data to provide insights into:
- **Flight Delay Patterns**: Understanding delay distributions and trends
- **Route Performance**: Identifying high-traffic routes and delay-prone connections
- **Airline Performance**: Comparing on-time performance across carriers
- **Temporal Analysis**: Examining delays by day of week, time of day, and season
- **Operational Insights**: Data-driven recommendations for operational improvements

## 📊 Features

- **Comprehensive Data Analysis**: Statistical analysis of flight delays, routes, and performance metrics
- **Multiple Visualization Types**: 
  - Static visualizations (matplotlib/seaborn)
  - Interactive dashboards (Plotly)
  - Heatmaps, bar charts, scatter plots, and time series
  - KPI dashboards and correlation matrices
- **Advanced Analytics**:
  - Correlation analysis between flight variables
  - Time series analysis and trend identification
  - Predictive modeling using machine learning (Random Forest)
  - Feature importance analysis
- **Real Data Integration**: Downloads and processes real aviation data from public sources
- **Professional Reports**: Automated generation of insights and recommendations
- **Jupyter Notebook**: Interactive notebook for detailed exploration and presentation

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this project
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Download Data** (first time only):
```bash
python data_downloader.py
```

This will:
- Download OpenFlights airport and route data
- Generate sample flight delay data for analysis

2. **Run Complete Analysis** (Recommended):
```bash
python run_analysis.py
```

Or run individual components:
```bash
# Basic analytics
python flight_analytics.py

# Advanced analytics (requires scikit-learn)
python advanced_analytics.py
```

3. **Interactive Jupyter Notebook**:
```bash
jupyter notebook aviation_analytics_notebook.ipynb
```

4. **Interactive Streamlit Dashboard**:
```bash
streamlit run dashboard.py
```

This will generate:
- `visualizations/delay_distribution.png` - Delay analysis charts
- `visualizations/route_analysis.png` - Route performance visualizations
- `visualizations/interactive_dashboard.html` - Interactive Plotly dashboard
- `visualizations/correlation_analysis.png` - Correlation heatmap
- `visualizations/time_series_analysis.png` - Time-based trend analysis
- `visualizations/comprehensive_dashboard.html` - Advanced interactive dashboard
- `visualizations/kpis.png` - Key performance indicators
- `aviation_analytics_report.txt` - Text report with insights

## 📸 Viewing Visualizations on GitHub

**PNG Images**: All PNG visualizations can be viewed directly on GitHub by clicking on them in the `visualizations/` folder. GitHub will display them inline.

**HTML Dashboards**: HTML files can be viewed by:
1. Clicking on the HTML file in GitHub
2. Clicking "Raw" to get the direct link
3. Copying the URL and pasting it into a service like [HTML Preview](https://htmlpreview.github.io/) or opening it directly in your browser

Alternatively, you can download the repository and open the HTML files locally in any web browser.

## 📁 Project Structure

```
flight/
├── data_downloader.py              # Downloads real aviation data
├── flight_analytics.py              # Basic analysis and visualization
├── advanced_analytics.py           # Advanced analytics with ML
├── run_analysis.py                  # Main script to run all analyses
├── aviation_analytics_notebook.ipynb # Jupyter notebook for exploration
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/                            # Data files (generated)
│   ├── airports.csv
│   ├── routes.csv
│   ├── airlines.csv
│   └── flight_delays_sample.csv
└── visualizations/                  # Generated visualizations
    ├── delay_distribution.png
    ├── route_analysis.png
    ├── interactive_dashboard.html
    ├── correlation_analysis.png
    ├── time_series_analysis.png
    ├── comprehensive_dashboard.html
    └── kpis.png
```

## 🔍 Data Sources

This project uses data from:
- **OpenFlights**: Airport, route, and airline data
- **Bureau of Transportation Statistics (BTS)**: Flight delay data (sample generated)
- **Sample Data Generator**: Creates realistic flight delay scenarios for demonstration

## 💡 Key Analytics Capabilities Demonstrated

1. **Data Processing**: 
   - Data cleaning and transformation
   - Handling missing values
   - Feature engineering
   - Data quality assessment

2. **Statistical Analysis**:
   - Descriptive statistics
   - Group aggregations
   - Trend analysis
   - Correlation analysis
   - Time series analysis

3. **Visualization**:
   - Static visualizations (matplotlib/seaborn)
   - Interactive dashboards (Plotly)
   - Multiple chart types (bar, line, scatter, heatmap, pie, histogram)
   - KPI dashboards
   - Correlation matrices

4. **Advanced Analytics**:
   - Machine learning models (Random Forest)
   - Predictive modeling for delay forecasting
   - Feature importance analysis
   - Model evaluation metrics

5. **Business Insights**:
   - Performance metrics
   - Comparative analysis
   - Actionable recommendations
   - Root cause analysis
