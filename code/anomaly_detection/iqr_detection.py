"""
Interquartile Range (IQR) based anomaly detection for multivariate data.
"""
import numpy as np
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def iqr_outlier_detection(df, col):
    """
    Detect outliers using the IQR method for a specific column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    col : str
        Column name to analyze
        
    Returns:
    --------
    pandas.Series
        Series with 1 for normal points, -1 for anomalies
    """
    # Calculate Q1, Q3 and IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define anomaly bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mark anomalies
    return df[col].apply(lambda x: -1 if x < lower_bound or x > upper_bound else 1)

def compute_anomaly_size(df, col):
    """
    Compute the anomaly magnitude for visualization purposes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    col : str
        Column name to analyze
        
    Returns:
    --------
    pandas.Series
        Series with anomaly magnitude values
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[col].apply(lambda x: abs(x - lower_bound) if x < lower_bound else 
                        (abs(x - upper_bound) if x > upper_bound else 0))

def preprocess_data(data_path):
    """
    Load and preprocess data with IQR anomaly detection.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
        
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

    # Apply IQR detection to each feature
    features = ['T/°C', 'W(NaCl)/%', 'W(Na2SO4)/%']
    df['Anomaly_T'] = iqr_outlier_detection(df, features[0])
    df['Anomaly_NaCl'] = iqr_outlier_detection(df, features[1])
    df['Anomaly_Na2SO4'] = iqr_outlier_detection(df, features[2])

    # Combine anomaly results - any feature anomaly makes the point anomalous
    df['Anomaly'] = df[['Anomaly_T', 'Anomaly_NaCl', 'Anomaly_Na2SO4']].min(axis=1)

    # Output results
    anomalies = df[df['Anomaly'] == -1]
    print(f"Anomaly detection results: {len(anomalies)} anomalies found")
    
    # Return cleaned data
    df_cleaned = df[df['Anomaly'] == 1].drop(columns=['Anomaly_T', 'Anomaly_NaCl', 
                                                     'Anomaly_Na2SO4', 'Anomaly'])
    df_cleaned.to_excel('data-cleaned-IQR.xlsx', index=False)
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_data = preprocess_data("data.xlsx")
    print(f"Cleaned data shape: {cleaned_data.shape}")