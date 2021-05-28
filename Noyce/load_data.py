
import pandas as pd
from utils.normalizer import normalize

def load_pol_data():
  
  df = pd.read_csv("./UCD_Noyce/Noyce/data/political_text/political_train.csv", encoding= 'unicode_escape')
  df_test = pd.read_csv("./UCD_Noyce/Noyce/data/political_text/political_test.csv", encoding= 'unicode_escape')
  df['text'] = df['text'].apply(normalize)
  df_test['text'] = df_test['text'].apply(normalize)
  return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_data(dset_name = 'political_text'):
  if dset_name == 'political_text':
    return load_pol_data()
  else:
    raise NameError('Dataset not known. Available Datasets: political_text')
    




  