import os
import json
import torch
import re
import uuid

from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .task_data import varlen_collate_fn

FILL_VAL = -1
class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, encoder_special_token_ids, 
                 decoder_special_token_ids, decoder_tokenizer, encoder_tokenizer,
                 data_dir, encoder_max_len, decoder_max_len, n_workers = 4,
                 use_sep =False, num_samples = 2500, extra_data=[], use_encoder=True):
        
        """
        Instantiate encoder-decoder training dataset
        """

        self.data_type = data_type
        self.use_encoder = use_encoder
        self.gen_token = gen_token
        self.use_sep = use_sep
        self.data_dir = data_dir
        self.n_workers = n_workers
        self.num_samples = num_samples
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_special_token_ids = encoder_special_token_ids
        self.decoder_special_token_ids = decoder_special_token_ids
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        if self.use_sep:
            self.dec_sep_token = decoder_special_token_ids["sep_token"]
        self.dec_ans_token = decoder_special_token_ids["ans_token"]
        self.dec_eos_token = decoder_special_token_ids["eos_token"]
        self.dec_pad_token = decoder_special_token_ids["pad_token"]

        if self.use_sep:
            self.enc_sep_token = encoder_special_token_ids["sep_token"]
        self.cls_token = encoder_special_token_ids["cls_token"]
        self.enc_ans_token = encoder_special_token_ids["ans_token"]
        self.enc_pad_token = encoder_special_token_ids["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            raw_ds = load_raw_samples(data_path,num_samples=self.num_samples)
            # with open(data_path, "r") as f:
            #     raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d

        self.data = []
        self.max_a_len = 0
        if len(data) > 0:
            self.data_tokenization(data)

        if len(extra_data) > 0:
            num_all_extra_data = len(extra_data)
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x: x, extra_data))
            self.data += extra_data

    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = " ".join([str(datum) for datum in data[1:]])
        try:
            if self.use_sep:
                dec_context, dec_qa = re.split(
                    str(self.decoder_special_token_ids["sep_token"]), data
                )
            else:
                dec_context = ""
                dec_qa = data
            dec_question, dec_answer = re.split(
                str(self.decoder_special_token_ids["ans_token"]), dec_qa
            )
            dec_context = [int(c) for c in dec_context.strip().split()]
            dec_question = [int(q) for q in dec_question.strip().split()]
            dec_answer = [
                int(a)
                for a in re.sub(
                    str(self.decoder_special_token_ids["eos_token"]), "", dec_answer
                )
                .strip()
                .split()
            ]
            if self.use_encoder:

                def dec_ids_to_enc_ids(dec_ids):
                    return self.encoder_tokenizer.encode(self.decoder_tokenizer.decode(dec_ids))

                enc_context, enc_question, enc_answer = map(
                    dec_ids_to_enc_ids, (dec_context, dec_question, dec_answer)
                )
                enc_examples = self.parse_enc_example(
                    self.cls_token, enc_context, enc_question, enc_answer
                )
                if enc_examples is None:
                    raise ValueError
            else:
                enc_examples = None
            dec_examples = self.parse_dec_example(
                gen_token, dec_context, dec_question, dec_answer
            )
            uid = uuid.uuid1().hex
        except ValueError:
            return
        return {
            "enc_examples": enc_examples,
            "dec_examples": dec_examples,
            "is_replay": True,
            "id": uid,
        }

    def enc_concat_example(self, cls_token, c, sep_token, q, ans_token, a):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > self.encoder_max_len:
            print(
                "an example with len {} is too long!".format(len(example) + 1)
            )
            raise ValueError
        example = cls_token + c[: self.encoder_max_len - len(example) - 1] + example
        return example

    def dec_concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > self.decoder_max_len:
            print(
                "an example with len {} is too long!".format(len(example) + 1)
            )
            raise ValueError
        example = (
            gen_token
            + c[: self.decoder_max_len - len(example) - 1]
            + example
            + eos_token
        )
        return example

    def parse_enc_example(self, cls_token, context, question, answer):
        if self.use_sep:
            cls_cq = self.enc_concat_example(
                [cls_token], context, [self.enc_sep_token], question, [], []
            )
        else:
            cls_cq = self.enc_concat_example([cls_token], context, [], question, [], [])
        return {"cls_cq": cls_cq}

    def parse_dec_example(self, gen_token, context, question, answer):
        if self.use_sep:
            cq_example = self.dec_concat_example(
                [gen_token],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                [],
                [],
            )
            cqa_example = self.dec_concat_example(
                [gen_token],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                answer,
                [],
            )
        else:
            cq_example = self.dec_concat_example(
                [gen_token], context, [], question, [self.dec_ans_token], [], []
            )
            cqa_example = self.dec_concat_example(
                [gen_token], context, [], question, [self.dec_ans_token], answer, []
            )
        qa_Y_example = self.dec_concat_example(
            [], [], [], [], [], answer, [self.dec_eos_token]
        )  # shifted
        qa_Y_example = [FILL_VAL] * (
            len(cqa_example) - len(qa_Y_example)
        ) + qa_Y_example
        if self.use_sep:
            gen_Y_example = self.dec_concat_example(
                [],
                context,
                [self.dec_sep_token],
                question,
                [self.dec_ans_token],
                answer,
                [self.dec_eos_token],
            )
        else:
            gen_Y_example = self.dec_concat_example(
                [],
                context,
                [],
                question,
                [self.dec_ans_token],
                answer,
                [self.dec_eos_token],
            )
        return {
            "cq": cq_example,
            "len_cq": len(cq_example),
            "cqa": cqa_example,
            "len_cqa": len(cqa_example),
            "qa_Y": qa_Y_example,
            "gen_Y": gen_Y_example,
        }

    def parallel_tokenization(self, d):
        examples = []
        dec_context = self.decoder_tokenizer.encode(d["context"])
        if self.use_encoder:
            enc_context = self.encoder_tokenizer.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            if self.use_encoder:
                enc_question = self.encoder_tokenizer.encode(qa["question"])
            dec_question = self.decoder_tokenizer.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            if self.use_encoder:
                enc_answer = []
            dec_answer = []
            for i, raw_answer in enumerate(raw_answers):
                if self.use_encoder:
                    enc_answer.extend(self.encoder_tokenizer.encode(raw_answer["text"]))
                dec_answer.extend(self.decoder_tokenizer.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    if self.use_encoder:
                        enc_answer.append(self.enc_pad_token)
                    dec_answer.append(self.dec_pad_token)
            max_a_len = max(max_a_len, len(dec_answer))

            if self.use_encoder:
                enc_examples = self.parse_enc_example(
                    self.cls_token, enc_context, enc_question, enc_answer
                )
            else:
                enc_examples = None

            dec_examples = self.parse_dec_example(
                self.gen_token, dec_context, dec_question, dec_answer
            )

            examples.append(
                {
                    "enc_examples": enc_examples,
                    "dec_examples": dec_examples,
                    "is_replay": False,
                    "id": qa.get("id", 0),
                }
            )

        return examples, max_a_len

    def data_tokenization(self, data):
        with Pool(self.n_workers) as pool:
            data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: x["dec_examples"]["len_cq"])
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x["id"])

    def get_indices(self):
        return [d["id"] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_raw_samples(json_file, num_samples=8000):

    # Load the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Extract the subset of data
    subset_data = []
    count = 0
    for element in data["data"]:
        if count >= num_samples:
            break
        
        for paragraph in element["paragraphs"]:
            if count >= num_samples:
                break
            
            subset_data.append(paragraph)
            count += len(paragraph["qas"])
            
            if count >= num_samples:
                # Trim the last paragraph to fit the desired number of samples
                extra_samples = count - num_samples
                paragraph["qas"] = paragraph["qas"][:-extra_samples]
                break

    # Create a new JSON object with the subset of data
    subset_json = {
        "version": data["version"],
        "data": [
            {
                "title": "Subset Dataset",
                "paragraphs": subset_data
            }
        ]
    }

    return subset_json