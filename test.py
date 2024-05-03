
import os
import torch
import json
import argparse

from tqdm import tqdm

from collections import OrderedDict
from models.model_utils import retrieve_task_map, generate_sequence, logits_to_tokens
from data_handling.task_data import create_dataloader
from data_handling.qa_dataset import QADataset
from data_handling.testing_utils import calculate_test_score
from pytorch_transformers import GPT2Tokenizer, BertTokenizer
from models.bert_encoder import BertModel, Encoder, BertConfig
from models.gpt2_decoder import GPT2LMHeadModel, Decoder, GPT2Config

encoder_special_tokens = {
    "cls_token": "[CLS]",
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "ans_token": "[ANS]",
}
decoder_special_tokens = {
    "ans_token": "__ans__",
    "pad_token": "__pad__",
    "unk_token": "__unk__",
    "eos_token": "<|endoftext|>",
    "gen_token": "__gen__",
}

def test_single_task(task, task_eval, decoder, encoder, score_dict, gen_token, encoder_special_token_ids, decoder_special_token_ids, decoder_config,
                     decoder_tokenizer, encoder_tokenizer, encoder_max_len, decoder_max_len, args):

    test_data = QADataset([TASK_MAP[task_eval]['test']], 'test', decoder_special_token_ids[task], encoder_special_token_ids, decoder_special_token_ids,
                          decoder_tokenizer,encoder_tokenizer,args.data_dir, encoder_max_len,decoder_max_len, use_encoder=True, num_samples=args.max_samples_per_task)
    max_len = test_data.max_a_len
    test_dataloader = create_dataloader(test_data, 'test', 2, 1, encoder_special_token_ids, decoder_special_token_ids)
    n_examples = len(test_data)
    samples_to_be_processed = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    max_total_lengths = [0 for _ in range(n_examples)]
    all_past_logits = [[0 for _ in range(n_examples)] for _ in range(decoder_config.n_layer)]
    decoder_max_len = decoder_config.n_positions

    cnt = 0
    for n_steps, (cls_cqs, cqs, len_cqs, _, _, _, _, _) in enumerate(test_dataloader):
        # assume n_gpus == 1
        if cls_cqs is not None:
            cls_cqs = cls_cqs
        cqs = cqs
        len_cqs = len_cqs
        n_inputs = cqs.shape[0]
        feature = encoder(cls_cqs.cuda())[0]

        all_outputs = decoder(input_ids=cqs.cuda(), feature=feature)
        outputs = all_outputs[0]
        pasts = all_outputs[1]
        next_logits = outputs[range(n_inputs), len_cqs - 1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_total_lengths[cnt] = max_len + test_data[cnt]["dec_examples"]["len_cq"]
            qa_results[cnt] = cqs[i][: len_cqs[i]]
            if next_tokens[i] != decoder_special_token_ids["eos_token"]:
                qa_results[cnt] = torch.cat((cqs[i][: len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) < min(max_total_lengths[cnt], decoder_max_len):
                    samples_to_be_processed.update([[cnt, None]])
                    for layer_id in range(decoder_config.n_layer):
                        all_past_logits[layer_id][cnt] = pasts[layer_id][
                            :, i, ..., : len_cqs[i], :
                        ].type(torch.float32)
            cnt += 1

    print("Generating responses...")
    generate_sequence(decoder, samples_to_be_processed, qa_results, all_past_logits, max_total_lengths,
                        decoder_config, decoder_special_token_ids, decoder_max_len, args)
    
    # Collate model answers along with ground-truth answers
    for i in range(len(test_data)):
        _, len_cq, _, _, qa_Y, _, = test_data[i]["dec_examples"].values()
        qa_Y = list(filter(lambda x: x != -1, qa_Y))[:-1]  # remove eos
        qa_Y = " ".join([str(y) for y in qa_Y]).split(str(decoder_special_token_ids["pad_token"]))
        qa_Y = [decoder_tokenizer.decode(list(map(int, y.split()))) for y in qa_Y]
        qa_results[i] = [decoder_tokenizer.decode(qa_results[i].tolist()[len_cq:]),qa_Y,]

    print("Scoring responses...")
    score = calculate_test_score(qa_results)
    score_dict[task] = score
    return decoder, score_dict
    

def test_task_in_combination(task, args):

    for ep in range(args.n_test_epochs):

        # Get tokenizer and config
        encoder_config = BertConfig.from_json_file(os.path.join(args.pretrained_model_dir, "encoder", "config.json"))
        decoder_config = GPT2Config.from_json_file(os.path.join(args.model_dir, "decoder", "config.json"))
        encoder_tokenizer = BertTokenizer.from_pretrained(os.path.join(args.pretrained_model_dir, "encoder","tokenizer"))
        decoder_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(args.pretrained_model_dir, "decoder", "tokenizer"))
        encoder_max_len = encoder_config.max_position_embeddings
        decoder_max_len = decoder_config.n_positions

        # Add special tokens to tokenizers
        encoder_tokenizer.add_tokens(list(encoder_special_tokens.values()))
        decoder_tokenizer.add_tokens(list(decoder_special_tokens.values()))
        encoder_special_token_ids = (
            {
                k: encoder_tokenizer.convert_tokens_to_ids(v)
                for k, v in encoder_special_tokens.items()
            }

        )
        decoder_special_token_ids = {
            k: decoder_tokenizer.convert_tokens_to_ids(v)
            for k, v in decoder_special_tokens.items()
        }

        gen_token = "__gen__"
        decoder_tokenizer.add_tokens([gen_token])
        decoder_special_tokens[task] = gen_token
        decoder_special_token_ids[task] = decoder_tokenizer.convert_tokens_to_ids(gen_token)

        encoder_model = BertModel(encoder_config)
        decoder_model = GPT2LMHeadModel(decoder_config, latent_as_gpt_emb=True, latent_as_gpt_attn=True)
        encoder = Encoder(encoder_model, encoder_tokenizer, feature_dim = args.feature_dim).cuda().eval()
        decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda().eval()

        decoder.resize_token_embeddings(len(decoder_tokenizer))

        # Load pre-trained weights
        encoder.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, "pretrained_encoder")))
        decoder.load_state_dict(torch.load(os.path.join(args.model_dir, "decoder", "fully_trained_decoder")))

        score_dict = {k: None for k in args.tasks}
        with torch.no_grad():
            for task_eval in args.tasks:
                test_single_task(task, task_eval, decoder, encoder, score_dict, gen_token, encoder_special_token_ids,decoder_special_token_ids, decoder_config,
                                 decoder_tokenizer, encoder_tokenizer, encoder_max_len, decoder_max_len,args)
        
        print(f"Scores: {score_dict}")

    with open(os.path.join(args.model_dir, "scores.json"), 'w') as f:
        json.dump(score_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--max_samples_per_task", type=int, default=2000)
    parser.add_argument("--n_test_epochs", type=int, default=9)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--tasks", nargs="+", default=["squad1"])
    parser.add_argument("--feature_dim", type=int, default=768)

    args = parser.parse_args()
    TASK_MAP = retrieve_task_map(args.data_dir)

    for task in args.tasks:
        test_task_in_combination(task, args)

