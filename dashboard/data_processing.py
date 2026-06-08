import pandas as pd
import numpy as np

def clean_and_feature_engineer(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.dropna(subset=['name', 'location', 'rest_type', 'cuisines'], inplace=True)
    
    # Clean Rate column
    def clean_rate(value):
        if pd.isna(value) or value == 'NEW' or value == '-':
            return np.nan
        else:
            value = str(value).split('/')[0]
            return float(value)
    
    df['rate'] = df['rate'].apply(clean_rate)
    df['rate'].fillna(df['rate'].mean(), inplace=True)
    
    # Clean Cost column
    def clean_cost(value):
        if pd.isna(value):
            return np.nan
        value = str(value).replace(',', '')
        return float(value)
    
    df['cost'] = df['approx_cost(for two people)'].apply(clean_cost)
    df['cost'].fillna(df['cost'].median(), inplace=True)
    
    # Feature Engineering
    
    # 1. Popularity Score: Weighted average of rating and votes
    # Normalizing votes first to scale it 0-1
    max_votes = df['votes'].max() if df['votes'].max() > 0 else 1
    df['norm_votes'] = df['votes'] / max_votes
    df['popularity_score'] = (df['rate'] * 0.7) + (df['norm_votes'] * 3.0) # Scale it roughly to 5
    
    # 2. Revenue Potential Score: based on cost and number of votes (proxy for traffic)
    df['revenue_potential'] = (df['cost'] * df['votes']) / (df['cost'].max() * df['votes'].max()) * 100
    
    # 3. Customer Attraction Score: High rating + Online Order + Table Booking
    df['online_order_val'] = df['online_order'].map({'Yes': 1, 'No': 0})
    df['book_table_val'] = df['book_table'].map({'Yes': 1, 'No': 0})
    df['attraction_score'] = (df['rate'] / 5.0) + (df['online_order_val'] * 0.2) + (df['book_table_val'] * 0.3)
    
    return df

if __name__ == "__main__":
    df = clean_and_feature_engineer('../data/zomato.csv')
    print("Initial processing complete.")
    print(df.head())
