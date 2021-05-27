import torch
from transformers import Trainer

class Custome_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids'].to(self.args.device)
        labels = inputs['labels'].to(self.args.device)
        outputs = self.model(input_ids)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits,labels)
        return (loss,{"logits": outputs.logits}) if return_outputs else loss

