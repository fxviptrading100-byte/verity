import numpy as np
import pandas as pd

def generate_dataset(n_samples=10000, seed=42):
    """
    Generates synthetic behavioral signal dataset
    for training the Verity verification model.
    Returns X (features) and y (labels: 1=human, 0=bot)
    """
    np.random.seed(seed)
    
    n_human = n_samples // 2
    n_bot = n_samples - n_human

    # Human behavioral signals
    human_data = np.column_stack([
        np.random.normal(187, 45, n_human),     # keystroke_gap_ms
        np.random.normal(272, 60, n_human),     # session_duration_s
        np.random.normal(21, 8, n_human),       # edit_count
        np.random.normal(0.82, 0.1, n_human),   # cursor_entropy
        np.random.normal(312, 80, n_human),     # content_length
        np.random.normal(0.91, 0.05, n_human),  # device_score
        np.random.normal(4.2, 1.1, n_human),    # typing_variance
        np.random.normal(0.78, 0.12, n_human),  # pause_ratio
    ])
    human_labels = np.ones(n_human)

    # Bot behavioral signals
    bot_data = np.column_stack([
        np.random.normal(12, 8, n_bot),         # near instant
        np.random.normal(3, 2, n_bot),          # very short session
        np.random.normal(1, 1, n_bot),          # almost no edits
        np.random.normal(0.15, 0.08, n_bot),    # low entropy
        np.random.normal(280, 60, n_bot),       # similar content length
        np.random.normal(0.3, 0.15, n_bot),     # low device score
        np.random.normal(0.1, 0.05, n_bot),     # no typing variance
        np.random.normal(0.05, 0.03, n_bot),    # no pauses
    ])
    bot_labels = np.zeros(n_bot)

    X = np.vstack([human_data, bot_data]).astype(np.float32)
    y = np.concatenate([human_labels, bot_labels]).astype(np.float32)

    # Clip negatives
    X = np.clip(X, 0, None)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

if __name__ == "__main__":
    X, y = generate_dataset(10000)
    print(f"Dataset generated: {X.shape}")
    print(f"Human samples: {int(y.sum())}")
    print(f"Bot samples: {int((1-y).sum())}")
    print(f"Feature means (human): {X[y==1].mean(axis=0).round(2)}")
    print(f"Feature means (bot): {X[y==0].mean(axis=0).round(2)}")