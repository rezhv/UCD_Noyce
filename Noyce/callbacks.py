import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(p):
    a = np.argmax(p.predictions, axis=1)
    b = p.label_ids
    acc = accuracy_score(a, b)
    return {'Accuracy': acc, 'Macro F1': f1_score(a, b, average='macro'),
            'Percision': precision_score(a, b), 'Recall': recall_score(a, b)}
