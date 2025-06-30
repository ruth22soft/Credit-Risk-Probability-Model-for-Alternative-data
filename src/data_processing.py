# src/data_processing.py
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# temporal features

def extract_temporal_features(df):
    df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    return df


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

def create_risk_labels(rfm_data):
    # Scale and cluster
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)
    rfm_data['is_high_risk'] = (rfm_data['Cluster'] == 0).astype(int)  # Critical line
    return rfm_data

if __name__ == "__main__":
    # 1. Load raw data
    raw_data = pd.read_csv("../data/raw/data.csv")
    
    # 2. Create RFM features
    rfm_data = create_rfm_features(raw_data)
    
    # 3. Add risk labels (THIS WAS MISSING!)
    rfm_with_risk = create_risk_labels(rfm_data)
    
    # 4. Save FINAL data (now includes 'is_high_risk')
    rfm_with_risk.to_csv("../data/processed/features.csv", index=True)

