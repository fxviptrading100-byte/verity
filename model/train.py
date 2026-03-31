import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\verity')

from env.data_generator import generate_dataset, normalize
from model.model import VerityNet

def train():
    print('Verity — Training human verification model')
    print('=' * 50)

    X, y = generate_dataset(n_samples=20000)
    X, mean, std = normalize(X)

    os.makedirs('model', exist_ok=True)
    np.save('model/mean.npy', mean)
    np.save('model/std.npy', std)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = VerityNet(input_dim=8, hidden_dims=[64, 32, 16], dropout=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_acc = ((val_pred >= 0.5).float() == y_val_t).float().mean().item()

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'input_dim': 8,
                'hidden_dims': [64, 32, 16]
            }, 'model/verity_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | Best: {best_val_acc:.2%}')

    print(f'Training complete. Best accuracy: {best_val_acc:.2%}')
    print('Model saved to model/verity_model.pt')
    return best_val_acc

if __name__ == '__main__':
    train()
