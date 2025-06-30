# src/data_processing.py
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_rfm_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',                                          # Frequency
        'Amount': 'sum'                                                    # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    return rfm

# temporal features

def extract_temporal_features(df):
    df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    return df

# save processed data

if __name__ == "__main__":
    raw_data = pd.read_csv("../data/raw/data.csv")
    processed_data = extract_temporal_features(raw_data)
    rfm_data = create_rfm_features(processed_data)
    rfm_data.to_csv("../data/processed/features.csv", index=True)

def create_risk_labels(rfm_data):
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
    
    # Cluster into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Label the least engaged cluster as high-risk (e.g., Cluster 0)
    rfm_data['is_high_risk'] = (rfm_data['Cluster'] == 0).astype(int)
    return rfm_data