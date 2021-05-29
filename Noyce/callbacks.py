import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import TrainerCallback
import torch

def compute_metrics(p):
    a = np.argmax(p.predictions, axis=1)
    b = p.label_ids
    acc = accuracy_score(a, b)
    return {'Accuracy': acc, 'Macro F1': f1_score(a, b, average='macro'),
            'Percision': precision_score(a, b), 'Recall': recall_score(a, b)}



class export_predictions_callback(TrainerCallback):

    def export_predictions(self, x,y, path = "./predictions.csv"):
        predictions_df = pd.DataFrame(data={"text": x, "prediction" : y})
        predictions_df.to_csv(path)

    def on_train_end(self, args, state, control, model = None, logs=None, **kwargs):
        model.eval()
        text = []
        predictions = []
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = model(batch['input_ids']).logits
                text = text + batch['text']
                predictions = predictions + torch.argmax(outputs, axis=1).cpu().numpy().tolist()

        self.export_predictions(text,predictions)