import argparse
from trainer import prepare_trainer


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--epochs", help="Number of Training Epochs", type=int, default=1)
  parser.add_argument("-b", "--batch_size", help="Batch Size", type=int, default=16)
  parser.add_argument("-l", "--logging_steps", help="Number of Steps to Run Evaluation", type=int, default=50)
  parser.add_argument("-lr", "--learning_rate", help="Value of Learning Rate", type=float, default=0.00002)
  parser.add_argument("-v", "--verbose", action="store_true", help="Whether or not to print warnings, states.", default=False)
  parser.add_argument("-ds", "--dataset", help="Path/Name of the Dataset to Train On", default= "political_text")
  parser.add_argument("-tl", "--tokenizationlength", help="Tokenization Max Length", default=128)
  parser.add_argument("-m", "--model", help="Name of the model: bert, xlmr, roberta", default="bert")
  parser.add_argument("-s", "--scheduler", action="store_true", help="Use Learning Rate Scheduler", default=False)
  parser.add_argument("-p", "--output_predictions", action="store_true", help="Output predictions as csv" ,default=False)
  parser.add_argument("-ac", "--accumulation_steps", help="Gradeient Accumulation Steps" ,default=1 , type = int)


  args = parser.parse_args()

  trainer = prepare_trainer(args)

  print("Training Started for", args.epochs, "epochs on", args.dataset)
  result = trainer.train()



  
      







