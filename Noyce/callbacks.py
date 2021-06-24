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
            'Percision': precision_score(a, b, average='macro'), 'Recall': recall_score(a, b, average='macro')}



class export_predictions_callback(TrainerCallback):

    def export_predictions(self, x,y, predictions,confidence, path = "./predictions.csv"):
        predictions_df = pd.DataFrame(data={"text": x, "prediction" : predictions, "labels": y, "confidence":confidence})
        predictions_df.to_csv(path)

    def on_train_end(self, args, state, control, model = None, tokenizer= None, eval_dataloader = None, logs=None, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        text = []
        predictions = []
        labels = []
        confidence = []
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = model(batch['input_ids'].to(device)).logits
                text = text + tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True)
                predictions = predictions + torch.argmax(outputs, axis=1).cpu().numpy().tolist()
                labels = labels + batch['labels'].cpu().numpy().tolist()
                confidence = confidence + torch.nn.functional.softmax(outputs,dim=1).cpu().numpy().tolist()

        confidence = [ ["{:0.2%}".format(x) for x in v] for v in confidence]

        self.export_predictions(text,labels,predictions, confidence)