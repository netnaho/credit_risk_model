# src/target_engineering.py

import pandas as pd
import numpy as np  # Import numpy directly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the is_high_risk target variable using RFM and K-Means clustering.

    Args:
        df: The raw transaction DataFrame.

    Returns:
        A DataFrame with CustomerId and the is_high_risk target column.
    """
    # --- 1. Calculate RFM Metrics ---

    # Ensure TransactionStartTime is a datetime object
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define a snapshot date for calculating recency
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Calculate RFM values for each customer
    rfm_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    })

    # Rename columns for clarity
    rfm_df.rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    }, inplace=True)

    # --- 2. Pre-process and Cluster Customers ---

    # Handle potential skewness and scale data before clustering
    # Use np.log1p which calculates log(1+x) to handle zero values gracefully
    rfm_log = np.log1p(rfm_df)

    # Standardize the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # Use K-Means to cluster customers into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # --- 3. Define and Assign the "High-Risk" Label ---

    # Analyze the cluster centroids to identify the high-risk group
    # We analyze the original scaled centroids for fair comparison
    cluster_analysis = pd.DataFrame(kmeans.cluster_centers_, columns=['Recency', 'Frequency', 'Monetary'])
    
    print("Cluster Analysis (Scaled Centroids):")
    print(cluster_analysis)

    # Identify the high-risk cluster: High Recency, Low Frequency, Low Monetary
    # In scaled terms: Highest Recency, Lowest Frequency, Lowest Monetary
    high_risk_cluster = cluster_analysis['Recency'].idxmax()
    
    print(f"\nIdentified High-Risk Cluster: {high_risk_cluster}")

    # Create the target variable
    rfm_df['is_high_risk'] = rfm_df['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)

    # --- 4. Return the final target DataFrame ---
    
    # We only need the customer ID and the target variable
    target_df = rfm_df[['is_high_risk']].reset_index()

    return target_df