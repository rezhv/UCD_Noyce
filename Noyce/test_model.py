import argparse
from models import Model
from noyce_tokenizer import Tokenizer

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelpath", help="Path to trained model")

  args = parser.parse_args()
  model = Model(path = args.modelpath)
  tokenizer = Tokenizer(path = args.modelpath)
  
  

