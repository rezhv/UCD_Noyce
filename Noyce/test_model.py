import argparse
from models import Model
from noyce_tokenizer import Tokenizer
from load_data import load_csv
import torch
import pandas as pd
from dataset import Dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelpath", help="Path to trained model")
  parser.add_argument("-d", "--datapath", help="Path to test data")

  args = parser.parse_args()
  model = Model(path = args.modelpath).to(device)
  tokenizer = Tokenizer(path = args.modelpath)
  x = load_csv(args.datapath)
  y = [0 for _ in range(len(x))]

  encodings = tokenizer(x, truncation=True, padding=True,
                                max_length=128,  return_tensors='pt')
  ds = Dataset(encodings, y)
  dl = DataLoader(ds, batch_size=32, shuffle=False)


  def export_predictions(x, predictions,confidence, path = "./predictions.csv"):
    predictions_df = pd.DataFrame(data={"text": x, "prediction" : predictions,"confidence":confidence})
    predictions_df.to_csv(path)

  model.eval()
  text = []
  predictions = []
  confidence = []
  with torch.no_grad():
      for batch in dl:
          outputs = model(batch['input_ids'].to(device)).logits
          text = text + tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True)
          predictions = predictions + torch.argmax(outputs, axis=1).cpu().numpy().tolist()
          confidence = confidence + torch.nn.functional.softmax(outputs,dim=1).cpu().numpy().tolist()

  confidence = [ ["{:0.2%}".format(x) for x in v] for v in confidence]

  export_predictions(text,predictions, confidence)


  

