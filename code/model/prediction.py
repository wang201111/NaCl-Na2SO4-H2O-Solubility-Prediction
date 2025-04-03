"""
Prediction script for using trained DENS-BO model.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# DNN model classes needed for loading the saved model
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

def predict(predict_file, model_path, in_cols, out_cols):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    predict_file : str
        Path to file with prediction data
    model_path : str
        Path to trained model and scalers
    in_cols : list
        Input column names
    out_cols : list
        Output column names
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with predictions
    """
    # Load scalers
    x_scaler = joblib.load(f"{model_path}/xScaler.pkl")
    y_scaler = joblib.load(f"{model_path}/yScaler.pkl")
    
    # Load column dictionary if exists
    col_dict_path = f"{model_path}/colDict.json"
    col_dict = {}
    if os.path.exists(col_dict_path):
        with open(col_dict_path, "r") as f:
            col_dict = json.loads(f.read())
    
    # Load data
    data = pd.read_excel(predict_file)
    data_copy = data.copy()
    
    # Preprocess data
    for col in data.columns:
        try:
            data[col] = data[col].astype(np.float64)
        except:
            data[col], index = pd.factorize(data[col])
            col_dict[col] = index.tolist()
    
    # Normalize data
    data[in_cols] = x_scaler.transform(data[in_cols].values)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(f"{model_path}/best.pth", map_location=device)
    
    # Extract model architecture from saved state
    if isinstance(model_info, dict) and 'model_config' in model_info:
        config = model_info['model_config']
        models = []
        for _ in range(config['num_models']):
            model = DNN(len(in_cols), len(out_cols), 
                        layer_dim=config['layer_dim'], 
                        node_dim=config['node_dim'])
            models.append(model)
        model = EnsembleDNN(models).to(device)
        model.load_state_dict(model_info['state_dict'])
    else:
        # Try to load with default configuration (backward compatibility)
        model = EnsembleDNN([DNN(len(in_cols), len(out_cols)) for _ in range(3)]).to(device)
        model.load_state_dict(model_info)
    
    # Make predictions
    model.eval()
    x = torch.FloatTensor(data[in_cols].values).to(device)
    with torch.no_grad():
        y_pred = model(x).cpu().numpy().reshape(-1, len(out_cols))
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred)
    
    # Ensure predictions are non-negative
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
    # Combine input features with predictions
    x_input = x_scaler.inverse_transform(data[in_cols].values)
    pred_cols = [f"{col} (predicted)" for col in out_cols]
    predicted_data = pd.DataFrame(x_input, columns=in_cols)
    predicted_data[pred_cols] = y_pred
    
    return predicted_data

if __name__ == "__main__":
    # Define parameters
    in_cols = ['T/°C', 'W(NaCl)/%']
    out_cols = ['W(Na2SO4)/%']
    model_path = "Static"
    predict_file = "predict.xlsx"
    
    # Make predictions
    results = predict(predict_file, model_path, in_cols, out_cols)
    results.to_excel("prediction_results.xlsx", index=False)