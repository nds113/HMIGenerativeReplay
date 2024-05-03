import os
import argparse
import torch

from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from pytorch_transformers import BertTokenizer, GPT2Tokenizer, AdamW

from models.hmi_module import HippocampalMemory, memorize_task_features
from models.model_utils import generate_replay_samples, sample_memory, retrieve_task_map
from models.bert_encoder import BertModel, Encoder, BertConfig
from models.gpt2_decoder import GPT2LMHeadModel, Decoder, GPT2Config
from data_handling.qa_dataset import QADataset
from data_handling.task_data import create_dataloader

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

def train(task_ids, encoder, decoder, memory_module, args):

    # Get tokenizer and config
    encoder_config = BertConfig.from_json_file(os.path.join(args.pretrained_model_dir, "encoder", "config.json"))
    decoder_config = GPT2Config.from_json_file(os.path.join(args.pretrained_model_dir, "decoder", "config.json"))
    encoder_tokenizer = BertTokenizer.from_pretrained(os.path.join(args.pretrained_model_dir, "encoder","tokenizer"))
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(args.pretrained_model_dir, "decoder", "tokenizer"))

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

    # Set up models if they're not already loaded
    if encoder is None or decoder is None:
        encoder_model = BertModel(encoder_config)
        decoder_model = GPT2LMHeadModel(decoder_config, latent_as_gpt_emb=True, latent_as_gpt_attn=True)
        encoder = Encoder(encoder_model, encoder_tokenizer, feature_dim = args.feature_dim).cuda()
        decoder = Decoder(decoder_model, feature_dim=args.feature_dim).cuda()

        # Load pre-trained weights
        encoder.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, "pretrained_encoder")))
        decoder.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, "pretrained_decoder")))

    encoder.resize_token_embeddings(len(encoder_tokenizer))
    decoder.resize_token_embeddings(len(decoder_tokenizer))

    if memory_module is None:
        # Instantiate hippocampal memory module
        memory_module = HippocampalMemory(args.feature_dim, max_size=args.max_memory_size)

    tasks = [args.tasks[task_id] for task_id in task_ids]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "encoder"),exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "decoder"),exist_ok=True)

    # Add special generation token for given task to decoder
    if args.add_task_tokens:
        gen_token = "__" + tasks[0] + "__"
    else:
        gen_token = "__gen__"
    decoder_tokenizer.add_tokens([gen_token])
    decoder_tokenizer.save_pretrained(args.model_dir)
    decoder_special_tokens[tasks[0]] = gen_token
    decoder_special_token_ids[tasks[0]] = decoder_tokenizer.convert_tokens_to_ids(gen_token)

    encoder_config.vocab_size = len(encoder_tokenizer)
    encoder_config.to_json_file(os.path.join(args.model_dir, "encoder", "config.json"))
    decoder_config.vocab_size = len(decoder_tokenizer)
    decoder_config.to_json_file(os.path.join(args.model_dir, "decoder", "config.json"))
    decoder.resize_token_embeddings(len(decoder_tokenizer))

    task_map = retrieve_task_map(args.data_dir, args.n_train_epochs)

    # Identify tasks we're training on
    train_dataset = [task_map[t]["train"] for t in tasks]

    os.makedirs(os.path.join(args.model_dir,"encoder", f"task_{tasks[0]}"), exist_ok=True)
    os.makedirs(os.path.join(args.model_dir,"decoder", f"task_{tasks[0]}"), exist_ok=True)
    encoder_max_len = encoder_config.max_position_embeddings
    decoder_max_len = decoder_config.n_positions

    # Generate replay data
    generated_replay_data = []
    if task_ids[0] > 0:
        prev_task = args.tasks[task_ids[0] - 1]
        with torch.no_grad():

            # Sample hippocampal memory module
            sampled_memory = sample_memory(memory_module, tasks[0], args)
            # Generate samples
            generate_replay_samples(tasks[0], decoder, generated_replay_data, decoder_special_token_ids,
                                    decoder_config, decoder_tokenizer, encoder_max_len, decoder_max_len,
                                    args, retrieved_memories=sampled_memory)

    train_qadata = QADataset(train_dataset, "train", decoder_special_token_ids[tasks[0]], encoder_special_token_ids,
                             decoder_special_token_ids,decoder_tokenizer,encoder_tokenizer,args.data_dir,
                             encoder_max_len, decoder_max_len,n_workers=args.n_workers, use_sep=args.use_sep,
                             extra_data=generated_replay_data, num_samples=args.max_samples_per_task)
    train_loader = create_dataloader(train_qadata, "train", args.train_batch_size, args.n_workers, encoder_special_token_ids, decoder_special_token_ids)
    n_train_epochs = task_map[tasks[0]]['n_train_epochs']

    # Setup up optimizer and learning schedule/rate
    params_to_optimize = list(decoder.named_parameters())
    if args.train_encoder:
        params_to_optimize += list(encoder.named_parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = False
    disable_weight_decay = ["bias","LayerNorm.bias", "LayerNorm.weight"]
    grouped_params = [{"params":[p for n,p in params_to_optimize if not any(nd in n for nd in disable_weight_decay)],"weight_decay":args.weight_decay},
                      {"params":[p for n,p in params_to_optimize if any(nd in n for nd in disable_weight_decay)],"weight_decay":0.0}]
    optimizer = AdamW(grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_func = CrossEntropyLoss(ignore_index=-1)

    if memory_module is not None:
        memory_module.set_new_task(tasks[0])
        memorize_task_features(train_qadata, encoder, memory_module, decoder_special_token_ids, encoder_special_token_ids)
        memory_module.quantize_memory()

    decoder.train()
    encoder.train() if args.train_encoder else encoder.eval()
    for epoch in range(n_train_epochs):
        total_gen_loss, total_qa_loss, total_num_inputs = 0, 0, 0
        for cls_cq, _, _, cqa, _, qa_Y, gen_Y, is_replay in tqdm(train_loader, desc=f"Training Epoch {epoch}"):

            cls_cq = cls_cq.cuda()
            cqa = cqa.cuda()
            qa_Y = qa_Y.cuda()
            gen_Y = gen_Y.cuda()

            # Encoder pass
            encoder_out = encoder(cls_cq)
            feature = memory_module.reconstruct_feature_from_memory(encoder_out[0].cpu()).cuda()

            logits = decoder(cqa, feature=feature)
            qa_loss = loss_func(logits[0].permute(0, 2, 1), qa_Y)
            gen_loss = loss_func(logits[0].permute(0, 2, 1), gen_Y)
            loss = torch.mean(qa_loss) + args.gen_lambda * torch.mean(gen_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Track progress
            num_inputs = sum(_cqa.shape[0] for _cqa in cqa)
            total_gen_loss += gen_loss.item()*num_inputs
            total_qa_loss += qa_loss.item()*num_inputs
            total_num_inputs += num_inputs

        print(f"Epoch {epoch}: Gen. Loss = {total_gen_loss/total_num_inputs}, QA Loss = {total_qa_loss/total_num_inputs}")

        # Save models
        if args.train_encoder:
            torch.save(encoder.state_dict(),os.path.join(args.model_dir,"encoder", f"task_{tasks[0]}", "enc_epoch_" + str(epoch + 1)))
        torch.save(decoder.state_dict(),os.path.join(args.model_dir,"decoder",f"task_{tasks[0]}", "dec_epoch_" + str(epoch + 1)))
        

    #Save final models after training
    if args.train_encoder:
        torch.save(encoder.state_dict(),os.path.join(args.model_dir,"encoder", f"task_{tasks[0]}", "trained_encoder"))
    torch.save(decoder.state_dict(),os.path.join(args.model_dir,"decoder",f"task_{tasks[0]}", "trained_decoder"))

    return encoder, decoder, memory_module


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)

    parser.add_argument(
        "--seq_train_type",
        type=str,
        default="hmi-lamol",
        choices=["hmi-lamol", "lamol"],
    )
    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--n_workers", type=int, default=4)

    # tasks
    parser.add_argument("--max_samples_per_task", type=int, default=2000)
    parser.add_argument("--tasks", nargs="+", default=["squad1"])

    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)

    #   batch size
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)

    parser.add_argument("--n_train_epochs", type=int, default=9)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--dynamic_epochs", action="store_true")

    #   others
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--tokens_weight", type=float, default=5)

    # decoding
    parser.add_argument("--top_k_qa", type=int, default=1)
    parser.add_argument("--top_p_qa", type=float, default=0.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)

    # for `hmi-lamol` and `lamol`
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--gen_lambda", type=float, default=0.25)

    # for `hmi-lamol`
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--max_memory_size", type=int, default=None)

    # for `lamol`
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.0)
    parser.add_argument("--temperature_lm", type=float, default=1.0)

    args = parser.parse_args()

    encoder, decoder, memory_module = None, None, None
    for task_id in range(len(args.tasks)):
        encoder, decoder, memory_module = train([task_id], encoder, decoder, memory_module, args)

    if args.train_encoder:
        torch.save(encoder.state_dict(),os.path.join(args.model_dir,"encoder", "fully_trained_encoder"))
    torch.save(decoder.state_dict(),os.path.join(args.model_dir,"decoder", "fully_trained_decoder"))