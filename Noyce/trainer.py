import torch
import transformers
from transformers import AdamW, Trainer

from callbacks import compute_metrics
from dataset import Dataset
from load_data import load_data
from models import Model
from noyce_tokenizer import Tokenizer


class Custome_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids'].to(self.args.device)
        labels = inputs['labels'].to(self.args.device)
        outputs = self.model(input_ids)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, {"logits": outputs.logits}) if return_outputs else loss


def prepare_trainer(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (not torch.cuda.is_available()):
        print("WARNING: TRAINING ON CPU. THIS WILL BE SLOW.")
    if (not args.verbose):
        transformers.logging.set_verbosity_error()

    x_train, y_train, x_test, y_test = load_data()
    tokenizer = Tokenizer(args.model)
    model = Model(args.model).to(device)
    train_encodings = tokenizer(x_train, truncation=True, padding=True,
                                max_length=args.tokenizationlength,  return_tensors='pt')
    test_encodings = tokenizer(x_test, truncation=True, padding=True,
                               max_length=args.tokenizationlength,  return_tensors='pt')
    train_set = Dataset(train_encodings, y_train)
    test_set = Dataset(test_encodings, y_test)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_args = transformers.TrainingArguments(logging_steps=args.logging_steps, output_dir="./",
                                                do_train=True,
                                                per_device_train_batch_size=args.batch_size,
                                                num_train_epochs=args.epochs,
                                                evaluation_strategy='steps',
                                                eval_steps=args.logging_steps,
                                                per_device_eval_batch_size=args.batch_size,
                                                logging_first_step=True,
                                                )

    trainer = Custome_Trainer(model=model,
                              args=train_args,
                              train_dataset=train_set,
                              optimizers=(optimizer, None),
                              eval_dataset=test_set,
                              compute_metrics=compute_metrics)

    trainer.remove_callback(transformers.PrinterCallback)

    return trainer
