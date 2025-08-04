# 🧭 Advanced Career Pathfinder Dashboard

An interactive Streamlit dashboard for analyzing data career paths using real job market data and Georgia Tech OMSA curriculum alignment.

## 🚀 Features

- **Custom Scoring System**: Adjust weights for different career factors
- **GT vs Market Alignment**: Compare OMSA courses with job requirements  
- **Skills Analysis**: Detailed skill demand and frequency analysis
- **Future Forecasting**: Career growth predictions and trend analysis
- **Transparency**: Full methodology explanations and data sources
- **PDF Reports**: Download custom analysis reports

## 📊 Data Sources

- **Job Market Data**: 286+ real job postings from Adzuna API
- **Course Data**: Georgia Tech OMSA curriculum analysis
- **Trend Data**: WEF, McKinsey, and OECD research reports

## 🔧 Requirements

- Python 3.8+
- Streamlit
- Plotly  
- ReportLab

## 🏃‍♂️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Streamlit Cloud Deployment

This app is ready for Streamlit Cloud deployment:

1. **Main script**: `app.py`
2. **Dependencies**: `requirements.txt`
3. **Data files**: All CSV files included

Deploy at: [share.streamlit.io](https://share.streamlit.io)

## 📁 Project Structure

```
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── gt_course_coverage_matrix.csv   # GT course analysis
├── gt_role_coverage_summary.csv    # Role coverage data
└── pathfinder/
    └── output/
        └── full_metrics_raw.csv    # Metrics data
```

## 🎯 Usage

1. **Select a section** from the sidebar
2. **Adjust weights** using the sliders (Custom Score Builder)
3. **Explore visualizations** and detailed breakdowns
4. **Download reports** for further analysis

## 🔒 Transparency

This dashboard includes full transparency features:
- Data source citations
- Methodology explanations  
- Limitation disclaimers
- Calculation formulas

## 📄 License

Educational use - Built for Georgia Tech OMSA program analysis.