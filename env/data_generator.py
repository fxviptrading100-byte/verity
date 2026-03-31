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

    # Human behavioral signals (much more overlap with bots)
    human_data = np.column_stack([
        np.random.normal(120, 100, n_human),     # keystroke_gap_ms (big overlap)
        np.random.normal(120, 150, n_human),     # session_duration_s (big overlap)
        np.random.normal(8, 10, n_human),        # edit_count (some humans edit very little)
        np.random.normal(0.45, 0.30, n_human),   # cursor_entropy (big overlap)
        np.random.normal(200, 180, n_human),     # content_length (big overlap)
        np.random.normal(0.60, 0.30, n_human),   # device_score (big overlap)
        np.random.normal(2.0, 2.5, n_human),     # typing_variance (big overlap)
        np.random.normal(0.40, 0.30, n_human),   # pause_ratio (big overlap)
    ])
    human_labels = np.ones(n_human)

    # Bot behavioral signals (much more human-like)
    bot_data = np.column_stack([
        np.random.normal(80, 80, n_bot),         # some bots type like humans
        np.random.normal(80, 100, n_bot),        # some bots have longer sessions
        np.random.normal(8, 8, n_bot),           # some bots edit like humans
        np.random.normal(0.40, 0.25, n_bot),     # some bots have human-like entropy
        np.random.normal(180, 150, n_bot),       # very similar content length
        np.random.normal(0.55, 0.30, n_bot),     # some bots have good device scores
        np.random.normal(2.5, 2.0, n_bot),       # some bots have human-like variance
        np.random.normal(0.35, 0.25, n_bot),     # some bots have human-like pauses
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