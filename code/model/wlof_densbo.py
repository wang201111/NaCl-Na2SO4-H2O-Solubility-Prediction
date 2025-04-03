import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Input and output features
IN_COLS = ['T/°C', 'W(NaCl)/%']
OUT_COLS = ['W(Na2SO4)/%']
TEST_RATE = 0.2


def weighted_lof(X, k=2):
    """
    Weighted Local Outlier Factor algorithm for anomaly detection
    
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


def preprocess_data(data_path):
    """
    Load and preprocess data with anomaly detection using WLOF
    
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

    # Apply weighted LOF
    X = df[['T/°C', 'W(NaCl)/%', 'W(Na2SO4)/%']].values
    df['LOF_Score'] = weighted_lof(X, k=2)

    # Identify anomalies
    threshold = df['LOF_Score'].quantile(0.8)
    df['Anomaly'] = df['LOF_Score'].apply(lambda x: -1 if x > threshold else 1)

    # Return cleaned data
    return df[df['Anomaly'] == 1].drop(columns=['LOF_Score', 'Anomaly'])


class FcBlock(nn.Module):
    """Fully connected block with batch normalization and activation"""
    def __init__(self, in_dim, out_dim):
        super(FcBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DNN(nn.Module):
    """Deep Neural Network architecture"""
    def __init__(self, in_dim, out_dim, layer_dim=3, node_dim=100):
        super(DNN, self).__init__()
        self.fc1 = FcBlock(in_dim, node_dim)
        self.fc_list = nn.ModuleList([FcBlock(node_dim, node_dim) for _ in range(layer_dim-2)])
        self.fcn = FcBlock(node_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        for fc in self.fc_list:
            x = fc(x)
        x = self.fcn(x)
        return x


class EnsembleDNN(nn.Module):
    """Ensemble of multiple DNN models"""
    def __init__(self, models):
        super(EnsembleDNN, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.mean(torch.stack(outputs), dim=0)


def create_ensemble(num_models, in_dim, out_dim, layer_dim, node_dim):
    """Create an ensemble of DNN models"""
    models = []
    for _ in range(num_models):
        model = DNN(in_dim, out_dim, layer_dim, node_dim)
        models.append(model)
    return models


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    # Mean Absolute Percentage Error
    relative_errors = [
        abs(y_true[i] - y_pred[i]) / y_true[i]
        for i in range(len(y_true)) if y_true[i] != 0
    ]
    
    if not relative_errors:
        return None, None
    
    average_relative_error = sum(relative_errors) / len(relative_errors)
    
    # Correlation coefficient (R²)
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_coefficient = correlation_matrix[0, 1]
    
    return average_relative_error, correlation_coefficient


def train_and_evaluate(params, cleaned_data):
    """
    Train and evaluate the ensemble DNN model
    
    Parameters:
    -----------
    params : tuple
        Hyperparameters (num_models, layer_dim, node_dim, learning_rate)
    cleaned_data : pandas.DataFrame
        Preprocessed data
        
    Returns:
    --------
    tuple
        Performance metrics (best_loss, average_relative_error, correlation_coefficient)
    """
    num_models, layer_dim, node_dim, lr = params

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize data
    x_scaler = StandardScaler().fit(cleaned_data[IN_COLS])
    y_scaler = StandardScaler().fit(cleaned_data[OUT_COLS])
    data = cleaned_data.copy()
    data[IN_COLS] = pd.DataFrame(x_scaler.transform(data[IN_COLS]), columns=IN_COLS)
    data[OUT_COLS] = pd.DataFrame(y_scaler.transform(data[OUT_COLS]), columns=OUT_COLS)
    
    # Save scalers
    os.makedirs("Static", exist_ok=True)
    joblib.dump(x_scaler, "Static/xScaler.pkl")
    joblib.dump(y_scaler, "Static/yScaler.pkl")

    # Create ensemble model
    models = create_ensemble(num_models, len(IN_COLS), len(OUT_COLS), layer_dim, node_dim)
    ensemble_dnn = EnsembleDNN(models).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ensemble_dnn.parameters(), lr=lr)

    # Split data into train and test sets
    index = int(data.shape[0] * TEST_RATE)
    test_data = data.iloc[:index].copy()
    train_data = data.iloc[index:].copy()
    x_test = test_data[IN_COLS]
    y_test = test_data[OUT_COLS]
    x_train = train_data[IN_COLS]
    y_train = train_data[OUT_COLS]

    # Training
    best_loss = float("inf")
    for epoch in tqdm(range(1000)):
        ensemble_dnn.train()
        x = torch.FloatTensor(x_train.values).to(device)
        y = torch.FloatTensor(y_train.values).to(device)
        y_pred = ensemble_dnn(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ensemble_dnn.eval()
        x = torch.FloatTensor(x_test.values).to(device)
        y = torch.FloatTensor(y_test.values).to(device)
        y_pred = ensemble_dnn(x)
        loss = criterion(y_pred, y)
        test_loss = loss.detach().cpu().numpy()

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(ensemble_dnn.state_dict(), 'Static/best.pth')

    # Load best model
    ensemble_dnn.load_state_dict(torch.load('Static/best.pth'))

    # Inverse transform predictions
    x = torch.FloatTensor(x_test.values).to(device)
    y_pred = ensemble_dnn(x).detach().cpu().numpy().reshape(-1, len(OUT_COLS))
    y_pred = y_scaler.inverse_transform(y_pred)
    y_true = y_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    avg_rel_error, r_squared = calculate_metrics(y_true.flatten(), y_pred.flatten())
    
    return best_loss, avg_rel_error, r_squared


def run_bayesian_optimization(cleaned_data, n_calls=50):
    """
    Run Bayesian Optimization to find the best hyperparameters
    
    Parameters:
    -----------
    cleaned_data : pandas.DataFrame
        Preprocessed data
    n_calls : int
        Number of optimization iterations
        
    Returns:
    --------
    dict
        Best parameters and optimization results
    """
    # Define parameter search space
    space = [
        Integer(1, 10, name='num_models'),     # Number of ensemble models
        Integer(1, 10, name='layer_dim'),      # Number of layers
        Integer(4, 256, name='node_dim'),      # Nodes per layer
        Real(1e-5, 5e-1, name='lr'),           # Learning rate
    ]

    # Results storage
    all_results = []
    
    # Objective function for optimization
    def objective(params):
        best_loss, avg_rel_error, r_squared = train_and_evaluate(params, cleaned_data)
        all_results.append({
            'num_models': params[0],
            'layer_dim': params[1],
            'node_dim': params[2],
            'lr': params[3],
            'best_loss': best_loss,
            'average_relative_error': avg_rel_error,
            'correlation_coefficient': r_squared
        })
        return best_loss
    
    # Run optimization
    res = gp_minimize(objective, space, n_calls=n_calls)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_excel('optimization_results.xlsx', index=False)
    
    return {
        'best_params': {
            'num_models': res.x[0],
            'layer_dim': res.x[1],
            'node_dim': res.x[2],
            'lr': res.x[3]
        },
        'best_loss': res.fun,
        'all_results': all_results
    }


def main():
    """Main execution function"""
    # Step 1: Preprocess data with WLOF anomaly detection
    cleaned_data = preprocess_data("data.xlsx")
    
    # Step 2: Run DENS-BO optimization
    results = run_bayesian_optimization(cleaned_data, n_calls=50)
    
    # Print results
    print("Best parameters:", results['best_params'])
    print(f"Best loss: {results['best_loss']:.4f}")


if __name__ == "__main__":
    main()