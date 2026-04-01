import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\verity')
from model.model import VerityNet

def load_cmu_dataset(filepath):
    print(f'Loading CMU dataset from {filepath}')
    df = pd.read_csv(filepath)
    print(f'Dataset shape: {df.shape}')
    print(f'Subjects: {df["subject"].nunique()}')

    feature_cols = [c for c in df.columns
                   if c.startswith('H.') or
                      c.startswith('DD.') or
                      c.startswith('UD.')]

    print(f'Feature columns: {len(feature_cols)}')

    samples = []
    labels = []
    subjects = df['subject'].unique()

    # Extract real human samples
    all_human_features = []
    for subject in subjects:
        subject_data = df[df['subject'] == subject][feature_cols].values
        for row in subject_data:
            row = row.astype(float)
            row = row[~np.isnan(row)]
            if len(row) < 10:
                continue
            hold_times = row[:10] * 1000
            dd_times = row[10:20] * 1000 if len(row) > 20 else row[:10] * 1000
            dwell_mean = float(np.mean(hold_times))
            dwell_std = float(np.std(hold_times))
            flight_mean = float(np.mean(dd_times))
            flight_std = float(np.std(dd_times))
            backspace_ratio = float(np.random.normal(0.06, 0.03))
            typing_variance = dwell_std
            pause_ratio = float(np.mean(hold_times > 150))
            features = np.array([
                dwell_mean,
                flight_mean,
                float(np.random.normal(8, 4)),
                min(dwell_std / 80, 1.0),
                10.0,
                0.85,
                typing_variance,
                pause_ratio
            ], dtype=np.float32)
            features = np.clip(features, 0, None)
            all_human_features.append(features)
            samples.append(features)
            labels.append(1.0)

    n_genuine = len(samples)
    print(f'Genuine human samples: {n_genuine}')

    # Compute human population statistics
    human_arr = np.array(all_human_features)
    pop_dwell_mean = float(human_arr[:, 0].mean())
    pop_dwell_std = float(human_arr[:, 0].std())
    pop_flight_mean = float(human_arr[:, 1].mean())
    pop_flight_std = float(human_arr[:, 1].std())

    print(f'Human population dwell: {pop_dwell_mean:.1f} +/- {pop_dwell_std:.1f}ms')
    print(f'Human population flight: {pop_flight_mean:.1f} +/- {pop_flight_std:.1f}ms')

    # Generate realistic bot samples
    # Bots are harder to detect — they try to mimic humans
    # Three types of bots:

    n_per_type = n_genuine // 3

    # Type 1: Naive bot — fast but slightly randomized
    for i in range(n_per_type):
        dwell = max(0, np.random.normal(25, 8))
        flight = max(0, np.random.normal(30, 12))
        bot = np.array([
            dwell,
            flight,
            0.0,
            dwell / 200,
            10.0,
            0.3,
            np.random.normal(3, 1),
            0.02
        ], dtype=np.float32)
        samples.append(np.clip(bot, 0, None))
        labels.append(0.0)

    # Type 2: Sophisticated bot — mimics human timing with less variance
    for i in range(n_per_type):
        dwell = max(0, np.random.normal(pop_dwell_mean * 0.7, pop_dwell_std * 0.15))
        flight = max(0, np.random.normal(pop_flight_mean * 0.6, pop_flight_std * 0.15))
        bot = np.array([
            dwell,
            flight,
            float(np.random.poisson(0.5)),
            min(abs(np.random.normal(0.2, 0.05)), 1.0),
            10.0,
            0.5,
            np.random.normal(pop_dwell_std * 0.2, 2),
            0.03
        ], dtype=np.float32)
        samples.append(np.clip(bot, 0, None))
        labels.append(0.0)

    # Type 3: Very sophisticated bot — close to human but no backspace, low variance
    remaining = n_genuine - (n_per_type * 2)
    for i in range(remaining):
        dwell = max(0, np.random.normal(pop_dwell_mean * 0.85, pop_dwell_std * 0.25))
        flight = max(0, np.random.normal(pop_flight_mean * 0.75, pop_flight_std * 0.25))
        bot = np.array([
            dwell,
            flight,
            float(np.random.poisson(1)),
            min(abs(np.random.normal(0.35, 0.08)), 1.0),
            10.0,
            0.6,
            np.random.normal(pop_dwell_std * 0.3, 3),
            float(np.random.uniform(0.01, 0.05))
        ], dtype=np.float32)
        samples.append(np.clip(bot, 0, None))
        labels.append(0.0)

    X = np.array(samples, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print(f'Total samples: {len(X)}')
    print(f'Human: {int(y.sum())} Bot: {int((1-y).sum())}')
    print(f'Human dwell mean: {X[y==1][:,0].mean():.2f}ms')
    print(f'Bot dwell mean: {X[y==0][:,0].mean():.2f}ms')

    return X, y

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def train_on_real_data():
    print('Verity - Training on REAL CMU Keystroke Dataset')
    print('=' * 55)

    data_path = 'data/DSL-StrongPasswordData.csv'
    if not os.path.exists(data_path):
        print(f'ERROR: Dataset not found at {data_path}')
        return

    X, y = load_cmu_dataset(data_path)
    X_norm, mean, std = normalize(X)

    np.save('model/mean.npy', mean)
    np.save('model/std.npy', std)
    print('Normalization params saved')

    split = int(0.8 * len(X))
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = VerityNet(input_dim=8, hidden_dims=[64, 32, 16], dropout=0.4)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)

    best_val_acc = 0
    best_epoch = 0
    epochs = 100

    print(f'Training on {len(X_train)} samples, validating on {len(X_val)}')
    print('-' * 55)

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

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'input_dim': 8,
                'hidden_dims': [64, 32, 16],
                'trained_on': 'CMU Keystroke Dynamics Benchmark Dataset',
                'n_subjects': 51,
                'n_samples': len(X),
                'bot_types': 3
            }, 'model/verity_model.pt')

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch+1:3d}/{epochs} | '
                f'Loss: {train_loss/len(train_loader):.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'Val Acc: {val_acc:.2%} | '
                f'Best: {best_val_acc:.2%} (ep {best_epoch+1})'
            )

    print(f'\nTraining complete.')
    print(f'Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch+1}')
    print(f'Model saved to model/verity_model.pt')
    print(f'Trained on REAL data from 51 human subjects')
    print(f'3 bot types: naive, sophisticated, very sophisticated')
    return best_val_acc

if __name__ == '__main__':
    train_on_real_data()
