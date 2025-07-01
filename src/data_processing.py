# src/data_processing.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# A. CUSTOM TRANSFORMER FOR TIME-BASED FEATURES
class TimeBasedFeatures(BaseEstimator, TransformerMixin):
    """
    Creates time-based features from the 'TransactionStartTime' column.
    """
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data,
        # so we just return self.
        return self
    
    def transform(self, X, y=None):
        # Make a copy to avoid changing the original DataFrame
        X_ = X.copy()
        
        # Convert 'TransactionStartTime' to datetime objects
        X_['TransactionStartTime'] = pd.to_datetime(X_['TransactionStartTime'])
        
        # Extract time-based features
        X_['TransactionHour'] = X_['TransactionStartTime'].dt.hour
        X_['TransactionDay'] = X_['TransactionStartTime'].dt.day
        X_['TransactionMonth'] = X_['TransactionStartTime'].dt.month
        X_['TransactionYear'] = X_['TransactionStartTime'].dt.year
        
        # Drop the original datetime column as it's no longer needed for the model
        X_ = X_.drop(columns=['TransactionStartTime'])
        
        return X_

# B. CUSTOM TRANSFORMER FOR AGGREGATE FEATURES
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates aggregated features for each customer.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Define aggregation logic
        agg_logic = {
            'Value': ['sum', 'mean', 'std', 'count'],
            'TransactionId': ['count'] # To get transaction frequency
        }
        
        # Group by CustomerId and apply aggregation
        agg_df = X_.groupby('CustomerId').agg(agg_logic)
        
        # Flatten the multi-level column names
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.rename(columns={'TransactionId_count': 'TransactionCount'}, inplace=True)
        
        # Reset index to turn CustomerId back into a column
        agg_df = agg_df.reset_index()
        
        return agg_df