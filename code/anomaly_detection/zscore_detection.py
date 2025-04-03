"""
Z-score based anomaly detection for multivariate data.
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def preprocess_data(data_path, threshold=2):
    """
    Load and preprocess data with Z-score anomaly detection.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    threshold : float
        Z-score threshold for anomaly detection
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    # Load data
    data = pd.read_excel(data_path)
    
    # Calculate Z-scores for all features
    z_scores = zscore(data)
    
    # Identify outliers (any feature with |z| > threshold)
    is_outlier = (np.abs(z_scores) > threshold).any(axis=1)
    
    # Split normal and anomalous data
    normal_data = data[~is_outlier].reset_index(drop=True)
    abnormal_data = data[is_outlier].reset_index(drop=True)
    
    # Calculate mean Z-scores for anomalies
    abnormal_z_scores = z_scores[is_outlier]
    abnormal_z_scores_mean = np.mean(abnormal_z_scores, axis=1)
    
    # Output results
    print(f"Anomaly detection results: {len(abnormal_data)} anomalies found")
    normal_data.to_excel('data-cleaned-Zscore.xlsx', index=False)
    
    return normal_data

if __name__ == "__main__":
    cleaned_data = preprocess_data("data.xlsx", threshold=2)
    print(f"Cleaned data shape: {cleaned_data.shape}")