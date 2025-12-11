import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# Load processed data
# ----------------------------
def load_processed_data(file_path: str) -> pd.DataFrame:
    """Load processed CSV data"""
    df = pd.read_csv(file_path)
    return df

# ----------------------------
# Calculate RFM metrics
# ----------------------------
def calculate_rfm(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary metrics per CustomerId
    """
    # Convert transaction times to datetime WITH timezone awareness
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    
    # Convert snapshot_date to datetime WITH timezone awareness
    snapshot_date = pd.to_datetime(snapshot_date, utc=True)
    
    # Group by CustomerId and calculate RFM
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    
    return rfm


# ----------------------------
# Cluster customers
# ----------------------------
def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Cluster customers using KMeans based on RFM metrics"""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

# ----------------------------
# Assign high-risk label
# ----------------------------
def assign_high_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 'is_high_risk' label to the cluster with highest risk
    (usually low frequency and low monetary value)
    """
    # Identify cluster with highest Recency and lowest Monetary + Frequency
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # The cluster with max Recency and min Frequency/Monetary = high risk
    high_risk_cluster = cluster_stats.sort_values(
        by=['Recency', 'Frequency', 'Monetary'],
        ascending=[False, True, True]
    ).iloc[0]['Cluster']
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm

# ----------------------------
# Merge high-risk label back into main dataset
# ----------------------------
def merge_risk_label(df: pd.DataFrame, rfm: pd.DataFrame, output_file: str) -> None:
    """
    Merge 'is_high_risk' into the main dataframe and save to CSV
    """
    df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    df.to_csv(output_file, index=False)
    print(f"Processed data with risk label saved to {output_file}")
