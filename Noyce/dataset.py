import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels, text):
        self.text = text
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {'input_ids': self.encodings['input_ids'][idx],
                'labels': torch.tensor(self.labels[idx]),
                'text': self.text[idx]}

    def __len__(self):
        return len(self.labels)
