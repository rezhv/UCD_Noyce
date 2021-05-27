from transformers import AutoTokenizer

def Tokenizer(name = 'bert'):
  tokenizers = {"bert" : "bert-base-uncased",
            "roberta" : "roberta-base",
            "xlmr" : "xlm-roberta-base"}

  return AutoTokenizer.from_pretrained(tokenizers[name])
