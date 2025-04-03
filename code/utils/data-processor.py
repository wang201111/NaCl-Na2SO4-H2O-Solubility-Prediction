"""
Data preprocessing utilities for solubility prediction models.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_excel(file_path)
        print(f"Successfully loaded data with shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_data(data):
    """
    Clean data by removing duplicates and handling missing values.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    dupes_removed = initial_rows - df.shape[0]
    print(f"Removed {dupes_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_removed = missing_before - df.isnull().sum().sum()
    print(f"Removed {missing_removed} rows with missing values")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def normalize_features(data, features, scaler=None):
    """
    Normalize features using StandardScaler.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    features : list
        List of feature column names
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-fitted scaler. If None, a new scaler will be created and fitted
        
    Returns:
    --------
    tuple
        (DataFrame with normalized features, fitted scaler)
    """
    df = data.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
    else:
        X = scaler.transform(df[features])
    
    for i, feature in enumerate(features):
        df[feature] = X[:, i]
    
    return df, scaler

def train_test_split(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    # Shuffle data
    data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Split
    split_idx = int(len(data) * test_size)
    test_data = data.iloc[:split_idx]
    train_data = data.iloc[split_idx:]
    
    print(f"Training set: {train_data.shape[0]} samples")
    print(f"Testing set: {test_data.shape[0]} samples")
    
    return train_data, test_data

def plot_3d_solubility(data, temp_col='T/°C', nacl_col='W(NaCl)/%', na2so4_col='W(Na2SO4)/%', 
                      save_path=None):
    """
    Create 3D visualization of solubility data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    temp_col : str
        Temperature column name
    nacl_col : str
        NaCl concentration column name
    na2so4_col : str
        Na2SO4 concentration column name
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(data[temp_col], data[nacl_col], data[na2so4_col], 
               c=data[temp_col], cmap='viridis', s=30, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(temp_col)
    ax.set_ylabel(nacl_col)
    ax.set_zlabel(na2so4_col)
    ax.set_title('3D Solubility Visualization')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Temperature (°C)')
    
    # Adjust view
    ax.view_init(elev=30, azim=45)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_outlier_comparison(original_data, iqr_data, zscore_data, wlof_data,
                               feature_col='W(Na2SO4)/%', save_path=None):
    """
    Create visualization comparing original data with data after outlier removal.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original dataset
    iqr_data : pandas.DataFrame
        Data after IQR outlier detection
    zscore_data : pandas.DataFrame
        Data after Z-score outlier detection
    wlof_data : pandas.DataFrame
        Data after WLOF outlier detection
    feature_col : str
        Feature to visualize
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot original data
    sns.histplot(original_data[feature_col], ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Original Data')
    
    # Plot IQR cleaned data
    sns.histplot(iqr_data[feature_col], ax=axes[0, 1], kde=True)
    axes[0, 1].set_title('IQR Cleaned Data')
    
    # Plot Z-score cleaned data
    sns.histplot(zscore_data[feature_col], ax=axes[1, 0], kde=True)
    axes[1, 0].set_title('Z-score Cleaned Data')
    
    # Plot WLOF cleaned data
    sns.histplot(wlof_data[feature_col], ax=axes[1, 1], kde=True)
    axes[1, 1].set_title('WLOF Cleaned Data')
    
    # Add overall title
    plt.suptitle(f'Distribution of {feature_col} After Different Outlier Detection Methods', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Example usage
    data = load_data("data/data.xlsx")
    cleaned_data = clean_data(data)
    
    # Visualize data
    plot_3d_solubility(cleaned_data, save_path="results/3d_solubility.png")
    
    # Split data
    train_data, test_data = train_test_split(cleaned_data)
    
    print("Data processing completed successfully")
