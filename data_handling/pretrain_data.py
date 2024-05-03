
import os
import pickle
import torch
import csv

from tqdm import tqdm

from datasets import load_dataset
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

def download_wiki40b_data(data_dir="./", wiki40b_type="test"):
    """Load data from source and save preprocessed data to file"""
    print("Downloading ...")
    
    dataset = load_dataset("wiki40b", "en", split=wiki40b_type)
    
    raw_text_data_path = os.path.join(data_dir, "wiki40b", "all_test_text.csv")
    os.makedirs(os.path.dirname(raw_text_data_path), exist_ok=True)
    
    with open(raw_text_data_path, "w", newline="") as f:
        writer = csv.writer(f)
        with tqdm(dataset, ncols=100) as pbar:
            for wiki in pbar:
                for text in (
                    wiki["text"]
                    .replace("_NEWLINE_", "\n_START_PARAGRAPH_\n")
                    .split("\n_START_")
                ):
                    key = "PARAGRAPH_\n"
                    if text[: len(key)] == key:
                        writer.writerow([text[len(key) :]])
    
    return raw_text_data_path


class PretrainDataset(Dataset):
    def __init__(self,
                 encoder_tokenizer,
                 decoder_tokenizer,
                 dec_tokens,
                 n_data_workers=1,
                 min_seq_len=1,
                 encoder_max_len=512,
                 decoder_max_len=512, 
                 data_path=None, 
                 data_type="train", 
                 source="wiki40b"):

        self.data_type = data_type
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.cls_token_id = self.encoder_tokenizer.cls_token_id
        self.gen_token_id = self.decoder_tokenizer.convert_tokens_to_ids(dec_tokens["gen_token"])
        self.eos_token_id = self.decoder_tokenizer.convert_tokens_to_ids(dec_tokens["eos_token"])
        self.n_data_workers = n_data_workers
        self.min_seq_len = min_seq_len
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        if os.path.exists(data_path) and not os.path.isdir(data_path):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            raw_text_path = download_wiki40b_data(data_path)
            raw_text_data = []
            with open(raw_text_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for text in reader:
                    raw_text_data.append(text[0].strip())
            data = self.tokenize_data(raw_text_data)
            new_data_path = os.path.splitext(raw_text_path)[0]+".pkl"
            with open(new_data_path, "wb") as f:
                pickle.dump(data, f)

        self.data = self.filter_by_seq_len(data)

    def tokenize_sample(self, text):
        enc_input_ids = [self.cls_token_id] + self.encoder_tokenizer.encode(text)
        dec_input_ids = (
            [self.gen_token_id] + self.decoder_tokenizer.encode(text) + [self.eos_token_id]
        )
        return enc_input_ids, dec_input_ids

    def tokenize_data(self, text_data):
        with Pool(self.n_data_workers) as pool:
            data = pool.map(self.tokenize_sample, text_data)
        return data

    def filter_by_seq_len(self, data):
        def len_filter(d):
            return (
                self.min_seq_len <= len(d[0]) <= self.encoder_max_len
                and self.min_seq_len <= len(d[1]) <= self.decoder_max_len
            )

        data = list(filter(len_filter, data))
        return data

    def __getitem__(self, index):
        encoder_ids, decoder_ids = self.data[index]
        return torch.LongTensor(encoder_ids), torch.LongTensor(decoder_ids)

    def __len__(self):
        return len(self.data)
    
def varlen_collate_fn(data, encoder_tokenizer=None, decoder_tokenizer=None,batch_size=4, enc_tokens=None, dec_tokens=None):
    enc_pad_id = encoder_tokenizer.convert_tokens_to_ids(enc_tokens["pad_token"])
    dec_pad_id = decoder_tokenizer.convert_tokens_to_ids(dec_tokens["pad_token"])

    enc_ids_list, dec_ids_list = zip(*data)

    enc_input_ids = pad_sequence(enc_ids_list, batch_first=True, padding_value=enc_pad_id)
    dec_input_ids = pad_sequence(dec_ids_list, batch_first=True, padding_value=dec_pad_id)
    dec_label_ids = pad_sequence(dec_ids_list, batch_first=True, padding_value=-1)

    return (
        enc_input_ids,
        dec_input_ids,
        dec_label_ids,
    )

def create_dataloader(dataset, batch_size, shuffle=False, data_workers=1, max_batch_size=10000, enc_tokens=None, dec_tokens=None,
                      encoder_tokenizer=None,decoder_tokenizer=None):

    def collate_fn(x, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, batch_size=batch_size, 
                   enc_tokens=enc_tokens, dec_tokens=dec_tokens):
        return varlen_collate_fn(x, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, batch_size=batch_size,
                                 enc_tokens=enc_tokens, dec_tokens=dec_tokens)

    batch_sampler = None
    dataloader = DataLoader(
        dataset,
        num_workers=data_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        batch_size=batch_size,
        batch_sampler=batch_sampler,
    )
    return dataloader
