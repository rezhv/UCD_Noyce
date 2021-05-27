from transformers import AutoModelForSequenceClassification

def Model(name = 'bert'):
  models = {"bert" : "bert-base-uncased",
            "roberta" : "roberta-base",
            "xlmr" : "xlm-roberta-base"}

  return AutoModelForSequenceClassification.from_pretrained(models[name])



