'''
import tarfile

def extract_data(path, target_path):
    with tarfile.open(path, 'r:gz') as tar:
        tar.extractall(path=target_path)
'''

import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from pathlib import Path


class thisDataset(Dataset):
    def __init__(self, csv_file, tokenizer, vocab, max_length=256):
        self.data = pd.read_csv(csv_file, header=None, names=['label', 'title', 'description'])
        self.data['text'] = self.data['title'] + ' ' + self.data['description']
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenizer(self.data.iloc[idx]['text'])
        text = [self.vocab[token] for token in text[:self.max_length]]
        label = int(self.data.iloc[idx]['label']) - 1
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(torch.tensor(_text, dtype=torch.long))
    label_list = torch.tensor(label_list, dtype=torch.long)
    text_list = pad_sequence(text_list, batch_first=True)
    return text_list, label_list





def yield_tokens(data):
    for text in data['text']:
        text = str(text)
        yield tokenizer(text)

'''
csv_file_path = '/content/drive/MyDrive/NLP/TextClassificationDatasets/ag_news/ag_news_csv/test.csv'


data = pd.read_csv(csv_file_path, header=None, names=['label', 'title', 'description'])

#train sampling size 115000
sampled_data = data.sample(n=7600, replace=False, random_state=42)


sampled_csv_path = '/content/drive/MyDrive/NLP/TextClassificationDatasets/ag_news/ag_news_csv/sampled_test.csv'


sampled_data.to_csv(sampled_csv_path, index=False)
'''

def vocabData(file_path):
  csv_train = Path(file_path)
  dataframe = pd.read_csv(csv_train, header=None, names=['label', 'title', 'description'])
  dataframe['text'] = dataframe['title'] + ' ' + dataframe['description']
  vocab = build_vocab_from_iterator(yield_tokens(dataframe), specials=["<unk>"])
  vocab.set_default_index(vocab["<unk>"])
  tokenizer = get_tokenizer('basic_english')

  return csv_train, tokenizer, vocab
'''
file_path ='/HMIGenerativeReplay/training_data/TextClassificationData/amazon/sampled_train.csv'
csv_train, tokenizer, vocab=vocabData(file_path)

train_dataset = thisDataset(csv_train, tokenizer, vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)

for text, label in train_loader:
    print(text[0], label[0])  
    break
'''