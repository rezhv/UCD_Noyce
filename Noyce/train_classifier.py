import argparse

import torch
import transformers
from transformers import AdamW

from callbacks import compute_metrics
from dataset import Dataset
from load_data import load_data
from models import Model
from noyce_tokenizer import Tokenizer
from trainer import Custome_Trainer


def prepare_trainer(args):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  x_train, y_train, x_test, y_test = load_data()
  tokenizer = Tokenizer()
  model = Model().to(device)
  train_encodings = tokenizer(x_train, truncation=True, padding=True,max_length =128,  return_tensors='pt')
  test_encodings = tokenizer(x_test, truncation=True, padding=True,max_length =128,  return_tensors='pt')
  train_set = Dataset(train_encodings, y_train)
  test_set = Dataset(test_encodings, y_test)
  optimizer = AdamW(model.parameters(), lr=args.learning_rate)

  args = transformers.TrainingArguments(logging_steps = args.logging_steps,output_dir="./",
                                        do_train=True,
                                        per_device_train_batch_size=args.batch_size,
                                        num_train_epochs= args.epochs,
                                        evaluation_strategy = 'steps',
                                        eval_steps= args.logging_steps,
                                        per_device_eval_batch_size=args.batch_size,
                                        )

  trainer = Custome_Trainer(model=model,
                            args=args,
                            train_dataset=train_set,
                            optimizers=(optimizer,None),
                            eval_dataset=test_set,
                            compute_metrics=compute_metrics)

  return trainer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--epochs", help="Number of Training Epochs", type=int, default=1)
  parser.add_argument("-b", "--batch_size", help="Batch Size", type=int, default=16)
  parser.add_argument("-l", "--logging_steps", help="Number of Steps to Run Evaluation", type=int, default=50)
  parser.add_argument("-r", "--learning_rate", help="Value of Learning Rate", type=float, default=2e-5)

  args = parser.parse_args()

  trainer = prepare_trainer(args)

  print("Training Initialized")
  trainer.train()
      







