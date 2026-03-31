import torch
import torch.nn as nn

class VerityNet(nn.Module):
    """
    PyTorch neural network for human vs bot classification.
    Input: 8-dimensional behavioral signal vector
    Output: probability of being human (0-1)
    """

    def __init__(self, input_dim=8, hidden_dims=[64, 32, 16], dropout=0.3):
        super(VerityNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            prob = self.forward(x)
            label = (prob >= 0.5).float()
            return label, prob


class VerityLSTM(nn.Module):
    """
    LSTM variant for sequential keystroke pattern analysis.
    Input: sequence of keystroke timings
    Output: probability of being human
    """

    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.3):
        super(VerityLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output).squeeze(-1)