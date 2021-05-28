import torch
import transformers
from transformers import AdamW
from callbacks import compute_metrics
from dataset import Dataset
from load_data import load_data
from models import Model
from noyce_tokenizer import Tokenizer
from trainer import Custome_Trainer


def prepare_trainer():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  x_train, y_train, x_test, y_test = load_data()
  tokenizer = Tokenizer()
  model = Model().to(device)
  train_encodings = tokenizer(x_train, truncation=True, padding=True,max_length =128,  return_tensors='pt')
  test_encodings = tokenizer(x_test, truncation=True, padding=True,max_length =128,  return_tensors='pt')
  train_set = Dataset(train_encodings, y_train)
  test_set = Dataset(test_encodings, y_test)
  optimizer = AdamW(model.parameters(), lr=2e-5)

  args = transformers.TrainingArguments(logging_steps = 50,output_dir="./",
                                        do_train=True,
                                        per_device_train_batch_size=16,
                                        num_train_epochs=5.0,
                                        evaluation_strategy = 'steps',
                                        eval_steps= 50,
                                        per_device_eval_batch_size=8,
                                        )

  trainer = Custome_Trainer(model=model,
                            args=args,
                            train_dataset=train_set,
                            optimizers=(optimizer,None),
                            eval_dataset=test_set,
                            compute_metrics=compute_metrics)

  return trainer

if __name__ == '__main__':
  trainer = prepare_trainer()
  trainer.train()
      







