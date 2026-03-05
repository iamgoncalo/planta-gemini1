import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch, nn = None, None

class F_Field_LSTM(nn.Module if nn else object):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        if nn is None: return
        super(F_Field_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class ForecasterPipeline:
    @staticmethod
    def prepare_data(df: pd.DataFrame, seq_length: int = 60) -> tuple:
        features = df[['T', 'CO2', 'Occ', 'Light', 'Power']].values
        targets = df['F_global'].values
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:(i + seq_length)])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    @staticmethod
    def train_model(model, X_train, y_train, epochs=50, lr=0.001):
        if torch is None: return model, []
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return model, losses
