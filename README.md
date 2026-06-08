# Zomato Restaurant Analytics & Revenue Intelligence Dashboard

## 🚀 Project Overview
This project is a comprehensive data analytics and machine learning solution designed for the restaurant industry. It leverages the Zomato dataset to provide actionable insights into restaurant performance, cuisine trends, and customer preferences. It includes a modern dashboard for visualization and a predictive model for restaurant success.

## 🛠 Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly, Seaborn
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest, Linear Regression)
- **Deployment Ready:** Python-based architecture

## 📂 Project Structure
- `data/`: CSV datasets
- `dashboard/`: Data processing and cleaning logic
- `models/`: Trained ML models and feature encoders
- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `reports/`: EDA and business insight reports

## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd zomato-dashboard
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Key Features
- **Overview Dashboard:** Instant KPIs on total restaurants, ratings, and costs.
- **Cuisine Analytics:** Deep dive into popular cuisines and their impact on ratings.
- **Location Intelligence:** Geographic distribution and density analysis.
- **ML Predictor:** Predict restaurant success (ratings) based on location, cost, and service type.
- **Business Insights:** AI-driven recommendations for new restaurant ventures.

## 📈 Data Pipeline
- Automated cleaning of missing values and duplicates.
- Feature engineering of success metrics: Popularity Score and Revenue Potential.
- Scalable ML training module for predictive analytics.
