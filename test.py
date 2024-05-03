
import os
import torch
import json
import argparse

from collections import OrderedDict
from models.model_utils import retrieve_task_map, generate_sequence, logits_to_tokens
from data_handling.task_data import create_dataloader
from data_handling.qa_dataset import QADataset
from pytorch_transformers import GPT2Tokenizer
from models.gpt2_decoder import GPT2LMHeadModel, Decoder, GPT2Config

def test_single_task(task, task_eval, decoder, score_dict):

    test_data = QADataset(task_map[task_eval]['test'], 'test', decoder_special_token_ids[task]).sort()
    max_len = test_data.max_a_len
    test_dataloader = create_dataloader(test_data, 'test')
    n_examples = len(test_data)
    samples_to_be_processed = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_past_logits = [[0 for _ in range(n_examples)] for _ in range(decoder_config.n_layer)]

    cnt = 0
    for sample_num, (cls_cqs, cqs, len_cqs, _, _, _, _, _) in enumerate(test_dataloader):

        num_inputs = cqs.shape[0]
        decoded = decoder(input_ids=cqs.cuda())
        decoded_output = decoded[0]
        past_logits = decoded[1]
        next_logits = decoded_output[range(num_inputs), len_cqs - 1, :]
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(num_inputs):
            max_total_lengths = max_len + test_data[cnt]['dec_examples']['len_cq']
            qa_results[cnt] = cqs[i][:len_cqs[i]]
            if next_tokens[i] != decoder_special_token_ids['eos_token']:
                qa_results[cnt] = torch.cat((cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) < min(max_total_lengths[cnt], args.decoder_max_len):
                    samples_to_be_processed.update([[cnt,None]])
                    for layer_id in range(decoder.n_layer):
                        all_past_logits[layer_id][cnt] = past_logits[layer_id][:, i, ..., : len_cqs[i], :].type(torch.float32)
            cnt += 1 

    generate_sequence(decoder, samples_to_be_processed, qa_results, all_past_logits, max_total_lengths,
                        decoder_config, decoder_special_token_ids, decoder_max_len, args)
    
    score = calculate_task_score(qa_results,
                                 dialogue_metric = "woz.en" in task,
                                 logical_metric = "wikisql" in task)
    score_dict[task] = score
    return decoder, score_dict
    

def test_task_in_combination(task):

    for ep in range():

        # Get tokenizer and config
        decoder_config = GPT2Config.from_json_file(os.path.join(args.model_dir, "decoder", "config.json"))
        decoder_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(args.model_dir, "decoder", "tokenizer"))

        # Add special tokens to tokenizers
        decoder_tokenizer.add_tokens(list(decoder_special_tokens.values()))
        decoder_special_token_ids = {
            k: decoder_tokenizer.convert_tokens_to_ids(v)
            for k, v in decoder_special_tokens.items()
        }

        gen_token = get_gen_token(task)
        decoder_tokenizer.add_tokens([gen_token])
        decoder_special_tokens[task] = gen_token
        decoder_special_token_ids[task] = decoder_tokenizer.convert_tokens_to_ids(gen_token)

        decoder_model = GPT2LMHeadModel(decoder_config, latent_as_gpt_emb=True, latent_as_gpt_attn=True)
        decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda()

        # Load pre-trained weights
        decoder.load_state_dict(torch.load(os.path.join(args.model_dir, "decoder", "fully_trained_decoder")))

        score_dict = {k: None for k in args.tasks}
        with torch.no_grad():
            for task_eval in args.tasks:
                test_single_task(task, task_eval, decoder, score_dict)
        
        print(f"Scores: {score_dict}")

    with open(os.path.join(args.model_dir, "scores.json"), 'w') as f:
        json.dump(score_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--max_samples_per_task", type=int, default=2000)
    parser.add_argument("--tasks", nargs="+", default=["squad1"])
    parser.add_argument("--feature_dim", type=int, default=768)

    args = parser.parse_args()
    TASK_MAP = retrieve_task_map(args.data_dir, args.n_train_epochs)

    for task in args.tasks:
        test_task_in_combination(task)

