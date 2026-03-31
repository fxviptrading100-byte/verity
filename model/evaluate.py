import torch
import numpy as np
import os
import sys

sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\verity')

from env.data_generator import generate_dataset, normalize
from model.model import VerityNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate():
    print('Verity — Grader Evaluation')
    print('=' * 50)

    checkpoint = torch.load('C:\\Users\\Admin\\Desktop\\verity\\model\\verity_model.pt', map_location='cpu')
    model = VerityNet(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    mean = np.load('C:\\Users\\Admin\\Desktop\\verity\\model\\mean.npy')
    std = np.load('C:\\Users\\Admin\\Desktop\\verity\\model\\std.npy')

    X_test, y_test = generate_dataset(n_samples=2000, seed=999)
    X_test_norm, _, _ = normalize(X_test, mean, std)
    X_test_t = torch.FloatTensor(X_test_norm)

    with torch.no_grad():
        probs = model(X_test_t).numpy()
        preds = (probs >= 0.5).astype(int)

    y_test_int = y_test.astype(int)

    acc = accuracy_score(y_test_int, preds)
    f1 = f1_score(y_test_int, preds)
    precision = precision_score(y_test_int, preds)
    recall = recall_score(y_test_int, preds)
    cm = confusion_matrix(y_test_int, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f'Accuracy:            {acc:.2%}')
    print(f'F1 Score:            {f1:.4f}')
    print(f'Precision:           {precision:.4f}')
    print(f'Recall:              {recall:.4f}')
    print(f'False Positive Rate: {fpr:.2%}')
    print(f'False Negative Rate: {fnr:.2%}')
    print(f'True Negatives:      {tn}')
    print(f'False Positives:     {fp}')
    print(f'False Negatives:     {fn}')
    print(f'True Positives:      {tp}')
    print(f'\n{classification_report(y_test_int, preds, target_names=["Bot", "Human"])}')

    score = (acc * 0.4) + (f1 * 0.3) + ((1-fpr) * 0.2) + ((1-fnr) * 0.1)
    print(f'Verity Score: {score:.4f} / 1.0000')

    return score

if __name__ == '__main__':
    evaluate()