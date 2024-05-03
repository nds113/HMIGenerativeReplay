import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

def load_gpt2_weights(model, config, gpt2_checkpoint_path):
    import re

    import numpy as np
    import tensorflow as tf
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                l = re.split(r"(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "w" or l[0] == "g":
                pointer = getattr(pointer, "weight")
            elif l[0] == "b":
                pointer = getattr(pointer, "bias")
            elif l[0] == "wpe" or l[0] == "wte":
                pointer = getattr(pointer, l[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)
    return model

def load_bert_weights(model, config, tf_checkpoint_path):

    import re

    import numpy as np
    import tensorflow as tf
    tf_path = os.path.abspath(tf_checkpoint_path)
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif l[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)
    return model

def sample_memory(memory_module, task, args):

    task_id = args.tasks.index(task)
    prev_tasks = args.tasks[:task_id]

    generation_size = dataset_map[task]['train']['data_size']
    generation_size = int(np.ceil(generation_size * args.gen_lm_sample_percentage)) // len(prev_tasks)

    samples = []
    for prev_task in prev_tasks:
        samples.append(memory_module.retrieve(prev_task, generation_size))
    samples = torch.cat(samples)

    return samples

def generate_replay_samples(task, model, generated_replay_data, decoder_special_token_ids, 
                            decoder_config, decoder_tokenizer, encoder_max_len, decoder_max_len,
                            args, retrieved_memories=None):

    # Determine number of replays to generate
    task_count = args.tasks.index(task)
    generation_size = dataset_map[task]['train']['data_size']
    generation_size = generation_size if generation_size < args.max_samples_per_task else args.max_samples_per_task
    generation_size = int(np.ceil(generation_size * args.gen_lm_sample_percentage))
    generation_size -= generation_size % task_count

    # Special task tokens to for each replay we plan to generate
    model.eval()
    qa_results = []
    for task_name in args.tasks[:task_count]:
        qa_results.extend([torch.tensor([decoder_special_token_ids[task_name]]) for _ in range(generation_size // task_count)])

    # Send retrieved memories to GPU
    if retrieved_memories is not None:
        retrieved_memories = [memory.cuda() for memory in retrieved_memories]

    all_pasts = [[torch.empty(2,decoder_config.n_head,0,decoder_config.n_embd // decoder_config.n_head, dtype=torch.float).cuda() for _ in range(generation_size)] for _ in range(decoder_config.n_layer)]
    max_len = (min(encoder_max_len, decoder_max_len))
    max_lengths = [max_len for _ in range(generation_size)]

    generation_samples = OrderedDict()
    for i in range(generation_size):
        generation_samples.update([[i, None]])
        
    generate_sequence(model, generation_samples, qa_results, all_pasts, 
                    max_lengths, decoder_config, decoder_special_token_ids,
                    decoder_max_len, args, features=retrieved_memories)

    model.train()
    qa_results=[result.tolist() for result in qa_results]
    generated_replay_data.extend(qa_results)
    qa_results = [decoder_tokenizer.decode(result) for result in qa_results]

def generate_sequence(model, generation_samples, qa_results, all_pasts, max_lengths, decoder_config,
                      decoder_special_token_ids, decoder_max_len, args, features=None, top_k=1):

    while len(generation_samples) > 0:
        first_id = next(iter(generation_samples))
        shortest_len = len(qa_results[first_id])

        #decode_batch_size = int(args.memory_sizes[0]* args.memory_factor[args.seq_train_type]// (shortest_len + 1) ** args.len_factor)
        decode_batch_size = 4
        it = iter(generation_samples)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, feature = [], [], []
            past = [[] for _ in range(decoder_config.n_layer)]

            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    input_ids.append(qa_results[cur_id][-1:])
                    for layer_id in range(decoder_config.n_layer):
                        past[layer_id].append(all_pasts[layer_id][cur_id])
                    if features is not None:
                        feature.append(features[cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            feature = torch.stack(feature) if feature else None
            for layer_id in range(decoder_config.n_layer):

                # Find maximum sequence length and hidden dimension size
                max_seq_length = max(p.size(2) for p in past[layer_id])
                max_hidden_size = max(p.size(3) for p in past[layer_id])

                # Pad each tensor to have the same max sequence length and hidden size
                past_padded = [F.pad(p, 
                                    (0, max_hidden_size - p.size(3),  # Right padding for hidden dimension
                                    0, max_seq_length - p.size(2)),  # Right padding for sequence length
                                    "constant", decoder_special_token_ids["pad_token"])
                            for p in past[layer_id]]

                # Stack the uniformly padded tensors
                past[layer_id] = torch.stack(past_padded, dim=1)

            all_outputs = model(input_ids=input_ids.cuda(), feature=feature, past=past)

            outputs = all_outputs[0]
            pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits, top_k=top_k).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == decoder_special_token_ids["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [
                        max_lengths[cur_id],
                        decoder_max_len,
                    ]:
                        remove_ids.append(cur_id)
                    else:
                        for layer_id in range(decoder_config.n_layer):
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(torch.float)

        for idx in remove_ids:
            remove_id(idx, generation_samples, all_pasts, decoder_config)
    

def remove_id(idx, generation_samples, all_pasts, decoder_config):
    assert idx in generation_samples
    del generation_samples[idx]
    for layer_id in range(decoder_config.n_layer):
        all_pasts[layer_id][idx] = 0

def logits_to_tokens(next_logits, top_k=1):
    if top_k == 1:
        # greedy decoding
        next_tokens = next_logits.argmax(-1, keepdim=True)
        return next_tokens
    
    # Apply top-K sampling
    top_k = min(top_k, next_logits.size(-1))
    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
    next_logits[indices_to_remove] = -float("Inf")
    
    log_probs = F.softmax(next_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens


def retrieve_task_map(data_dir, n_train_epochs=9):

    task_map = {

        "squad1": {
            "train": os.path.join(data_dir, "squad-train-v1.1.json"),
            "eval": os.path.join(data_dir, "squad-dev-v1.1.json"),
            "test": os.path.join(data_dir, "squad-dev-v1.1.json"),
            "n_train_epochs": n_train_epochs,
        },
        "squad2": {
            "train": os.path.join(data_dir, "squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "squad-dev-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "iwslt.en.de": {
            "train": os.path.join(data_dir, "iwslt.en.de_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "iwslt.en.de_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "iwslt.en.de_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "cnn_dailymail": {
            "train": os.path.join(data_dir, "cnn_dailymail_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "cnn_dailymail_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "cnn_dailymail_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "multinli.in.out": {
            "train": os.path.join(
                data_dir, "multinli.in.out_to_squad-train-v2.0.json"
            ),
            "eval": os.path.join(data_dir, "multinli.in.out_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "multinli.in.out_to_squad-dev-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "sst": {
            "train": os.path.join(data_dir, "sst_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "sst_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "sst_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "srl": {
            "train": os.path.join(data_dir, "srl_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "srl_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "srl_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "zre": {
            "train": os.path.join(data_dir, "zre_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "zre_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "zre_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "woz.en": {
            "train": os.path.join(data_dir, "woz.en_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "woz.en_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "woz.en_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "wikisql": {
            "train": os.path.join(data_dir, "wikisql_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "wikisql_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "wikisql_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "schema": {
            "train": os.path.join(data_dir, "schema_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "schema_to_squad-dev-v2.0.json"),
            "test": os.path.join(data_dir, "schema_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "ag": {
            "train": os.path.join(data_dir, "ag_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "ag_to_squad-test-v2.0.json"),
            "test": os.path.join(data_dir, "ag_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "dbpedia": {
            "train": os.path.join(data_dir, "dbpedia_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "dbpedia_to_squad-test-v2.0.json"),
            "test": os.path.join(data_dir, "dbpedia_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "yahoo": {
            "train": os.path.join(data_dir, "yahoo_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "yahoo_to_squad-test-v2.0.json"),
            "test": os.path.join(data_dir, "yahoo_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "amazon": {
            "train": os.path.join(data_dir, "amazon_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "amazon_to_squad-test-v2.0.json"),
            "test": os.path.join(data_dir, "amazon_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
        "yelp": {
            "train": os.path.join(data_dir, "yelp_to_squad-train-v2.0.json"),
            "eval": os.path.join(data_dir, "yelp_to_squad-test-v2.0.json"),
            "test": os.path.join(data_dir, "yelp_to_squad-test-v2.0.json"),
            "n_train_epochs": n_train_epochs,
        },
    }

    return task_map

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

