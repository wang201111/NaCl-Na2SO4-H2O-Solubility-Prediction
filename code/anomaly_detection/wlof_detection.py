"""
Weighted Local Outlier Factor (WLOF) based anomaly detection for multivariate data.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def weighted_lof(X, k=2):
    """
    Weighted Local Outlier Factor algorithm for anomaly detection.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data with features
    k : int
        Number of neighbors
        
    Returns:
    --------
    numpy.ndarray
        LOF scores for each data point
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Calculate local density for each point
    local_density = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        weights = 1 / distances[i, 1:]  # Inverse distance as weight
        local_density[i] = np.sum(weights)

    # Calculate LOF scores
    lof_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        lof_scores[i] = np.sum(local_density[indices[i, 1:]] / local_density[i]) / k

    return lof_scores

def preprocess_data(data_path, quantile=0.8):
    """
    Load and preprocess data with WLOF anomaly detection.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    quantile : float
        Quantile threshold for anomaly detection
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    # Load data
    data = pd.read_excel(data_path)
    data = data.sample(frac=1.0)
    df = pd.DataFrame(data)
    df = df.drop_duplicates()

    # Extract features
    X = df[['T/°C', 'W(NaCl)/%', 'W(Na2SO4)/%']].values

    # Calculate WLOF scores
    df['LOF_Score'] = weighted_lof(X, k=2)

    # Determine anomaly threshold
    threshold = df['LOF_Score'].quantile(quantile)
    df['Anomaly'] = df['LOF_Score'].apply(lambda x: -1 if x > threshold else 1)

    # Output results
    anomalies = df[df['Anomaly'] == -1]
    print(f"Anomaly detection results: {len(anomalies)} anomalies found")
    
    # Return cleaned data
    df_cleaned = df[df['Anomaly'] == 1].drop(columns=['LOF_Score', 'Anomaly'])
    df_cleaned.to_excel('data-cleaned-WLOF.xlsx', index=False)
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_data = preprocess_data("data.xlsx")
    print(f"Cleaned data shape: {cleaned_data.shape}")