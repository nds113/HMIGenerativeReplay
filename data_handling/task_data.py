import os
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

def pad_to_max_len(seq, pad_len, val):
    return seq + [val] * pad_len


def pad_all_to_max_len(batch, val):
    max_len = max(len(seq) for seq in batch)
    return [pad_to_max_len(seq, max_len - len(seq), val) for seq in batch]

def varlen_collate_fn(data, decoder_special_token_ids=None, encoder_special_token_ids=None, n_gpus=1):

    cqs = torch.tensor(pad_all_to_max_len([datum["dec_examples"]["cq"] for datum in data],decoder_special_token_ids["pad_token"],))
    len_cqs = torch.tensor([datum["dec_examples"]["len_cq"] for datum in data])
    cqas = torch.tensor(pad_all_to_max_len([datum["dec_examples"]["cqa"] for datum in data],decoder_special_token_ids["pad_token"],))
    len_cqas = torch.tensor([datum["dec_examples"]["len_cqa"] for datum in data])
    qa_Ys = torch.tensor(pad_all_to_max_len([datum["dec_examples"]["qa_Y"] for datum in data], -1))
    gen_Ys = torch.tensor(pad_all_to_max_len([datum["dec_examples"]["gen_Y"] for datum in data], -1))
    is_replays = torch.tensor([datum["is_replay"] for datum in data])

    cls_cq = None
    if data[0]["enc_examples"] is not None:
        cls_cq = torch.tensor(pad_all_to_max_len([datum["enc_examples"]["cls_cq"] for datum in data],encoder_special_token_ids["pad_token"],))
    return (
        cls_cq,
        cqs,
        len_cqs,
        cqas,
        len_cqas,
        qa_Ys,
        gen_Ys,
        is_replays,
    )

def create_dataloader(dataset, data_type, batch_size, n_workers, encoder_special_token_ids, decoder_special_token_ids, max_batch_size=16):

    def collate_fn(x, encoder_special_token_ids=encoder_special_token_ids, decoder_special_token_ids=decoder_special_token_ids):
        return varlen_collate_fn(x, decoder_special_token_ids=decoder_special_token_ids, encoder_special_token_ids=encoder_special_token_ids)

    shuffle = not (data_type != "train")
    batch_sampler = None

    dataloader = DataLoader(
        dataset,
        num_workers=n_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        batch_size=batch_size,
        batch_sampler=batch_sampler,
    )
    return dataloader

dataset_map = {
        "squad1": {
            "train": {
                "data_size": 87599,
                "max_a_len": 77
            },
            "eval": {
                "data_size": 34726,
                "max_a_len": 70
            },
            "test": {
                "data_size": 34726,
                "max_a_len": 70
            }
        },
        "squad2": {
            "train": {
                "data_size": 130319,
                "max_a_len": 77
            },
            "eval": {
                "data_size": 26247,
                "max_a_len": 41
            },
            "test": {
                "data_size": 26247,
                "max_a_len": 41
            }
        },
        "iwslt.en.de": {
            "train": {
                "data_size": 196884,
                "max_a_len": 1396
            },
            "eval": {
                "data_size": 993,
                "max_a_len": 235
            },
            "test": {
                "data_size": 1305,
                "max_a_len": 187
            }
        },
        "cnn_dailymail": {
            "train": {
                "data_size": 287227,
                "max_a_len": 2688
            },
            "eval": {
                "data_size": 13368,
                "max_a_len": 2123
            },
            "test": {
                "data_size": 11490,
                "max_a_len": 870
            }
        },
        "multinli.in.out": {
            "train": {
                "data_size": 392702,
                "max_a_len": 3
            },
            "eval": {
                "data_size": 20000,
                "max_a_len": 3
            },
            "test": {
                "data_size": 19643,
                "max_a_len": 1
            }
        },
        "sst": {
            "train": {
                "data_size": 6920,
                "max_a_len": 1
            },
            "eval": {
                "data_size": 872,
                "max_a_len": 1
            },
            "test": {
                "data_size": 1821,
                "max_a_len": 1
            }
        },
        "srl": {
            "train": {
                "data_size": 6414,
                "max_a_len": 65
            },
            "eval": {
                "data_size": 2183,
                "max_a_len": 64
            },
            "test": {
                "data_size": 2201,
                "max_a_len": 54
            }
        },
        "zre": {
            "train": {
                "data_size": 840000,
                "max_a_len": 123
            },
            "eval": {
                "data_size": 600,
                "max_a_len": 13
            },
            "test": {
                "data_size": 12000,
                "max_a_len": 24
            }
        },
        "woz.en": {
            "train": {
                "data_size": 2536,
                "max_a_len": 17
            },
            "eval": {
                "data_size": 830,
                "max_a_len": 17
            },
            "test": {
                "data_size": 1646,
                "max_a_len": 16
            }
        },
        "wikisql": {
            "train": {
                "data_size": 56355,
                "max_a_len": 157
            },
            "eval": {
                "data_size": 8421,
                "max_a_len": 82
            },
            "test": {
                "data_size": 15878,
                "max_a_len": 85
            }
        },
        "schema": {
            "train": {
                "data_size": 80,
                "max_a_len": 4
            },
            "eval": {
                "data_size": 82,
                "max_a_len": 3
            },
            "test": {
                "data_size": 100,
                "max_a_len": 4
            }
        },
        "ag": {
            "train": {
                "data_size": 115000,
                "max_a_len": 4
            },
            "eval": {
                "data_size": 7600,
                "max_a_len": 4
            },
            "test": {
                "data_size": 7600,
                "max_a_len": 4
            }
        },
        "dbpedia": {
            "train": {
                "data_size": 115000,
                "max_a_len": 6
            },
            "eval": {
                "data_size": 7600,
                "max_a_len": 6
            },
            "test": {
                "data_size": 7600,
                "max_a_len": 6
            }
        },
        "yahoo": {
            "train": {
                "data_size": 115000,
                "max_a_len": 4
            },
            "eval": {
                "data_size": 7600,
                "max_a_len": 4
            },
            "test": {
                "data_size": 7600,
                "max_a_len": 4
            }
        },
        "amazon": {
            "train": {
                "data_size": 115000,
                "max_a_len": 2
            },
            "eval": {
                "data_size": 7600,
                "max_a_len": 2
            },
            "test": {
                "data_size": 7600,
                "max_a_len": 2
            }
        },
        "yelp": {
            "train": {
                "data_size": 115000,
                "max_a_len": 2
            },
            "eval": {
                "data_size": 7600,
                "max_a_len": 2
            },
            "test": {
                "data_size": 7600,
                "max_a_len": 2
            }
        }
    }

