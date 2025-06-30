# src/data_processing.py
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



# temporal features

def extract_temporal_features(df):
    """Extract time-based features from TransactionStartTime"""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    
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

def encode_categorical(df, cols_to_encode):
    """One-hot encode categorical features"""
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded = encoder.fit_transform(df[cols_to_encode])
    
    # Create feature names
    feature_names = []
    for col, categories in zip(cols_to_encode, encoder.categories_):
        feature_names += [f"{col}_{cat}" for cat in categories[1:]]
    
    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded, columns=feature_names)
    
    # Combine with original data
    return pd.concat([df.drop(cols_to_encode, axis=1), encoded_df], axis=1)

# src/data_processing.py
def main():
    # 1. Load raw data
    raw_data = pd.read_csv("../data/raw/data.csv")
    
    # 2. Extract temporal features (NEW)
    data_with_time = extract_temporal_features(raw_data)
    
    # 3. Create RFM features
    rfm_data = create_rfm_features(data_with_time)
    
    # 4. Add risk labels
    rfm_with_risk = create_risk_labels(rfm_data)
    
    # 5. Encode categoricals (NEW)
    categorical_cols = ['ProductCategory', 'ChannelId']  # Adjust based on your data
    final_data = encode_categorical(rfm_with_risk, categorical_cols)
    
    # 6. Save
    final_data.to_csv("../data/processed/features.csv", index=False)

