import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# Adding parent dir to path to import data_processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dashboard'))
from data_processing import clean_and_feature_engineer

def train_models(data_path):
    df = clean_and_feature_engineer(data_path)
    
    # Selecting features for ML
    cols = ['online_order', 'book_table', 'location', 'rest_type', 'cost', 'votes']
    target = 'rate'
    
    X = df[cols].copy()
    y = df[target]
    
    # Encoding categorical variables
    le = LabelEncoder()
    cat_cols = ['online_order', 'book_table', 'location', 'rest_type']
    
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        # Save encoder for future use
        joblib.dump(le, f'models/{col}_encoder.pkl')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # Save the best model (using RF as default winner for this type of data)
    joblib.dump(rf, 'models/restaurant_success_model.pkl')
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': cols,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print(f"Linear Regression R2: {lr_r2}")
    print(f"Random Forest R2: {rf_r2}")
    
    return rf_r2, lr_r2

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    train_models('data/zomato.csv')
