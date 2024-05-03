import argparse
import os
import random
import torch
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from pytorch_transformers import BertTokenizer, GPT2Tokenizer, AdamW, CONFIG_NAME

from data_handling.pretrain_data import PretrainDataset, create_dataloader
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

def pretrain_model(pretrain_args):

    os.makedirs(os.path.join(pretrain_args.model_dir, "encoder"), exist_ok=True)
    os.makedirs(os.path.join(pretrain_args.model_dir, "decoder"), exist_ok=True)

    # Load configurations
    decoder_config = GPT2Config.from_pretrained("gpt2")
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")

    # Load tokenizers
    decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Construct encoder-decoder
    encoder_model = BertModel.from_pretrained("bert-base-uncased")
    decoder_model = GPT2LMHeadModel.from_pretrained("gpt2", config=decoder_config, latent_as_gpt_emb=True, latent_as_gpt_attn=True)
    encoder = Encoder(encoder_model, encoder_tokenizer, feature_dim = pretrain_args.feature_dim).cuda()
    decoder = Decoder(decoder_model, feature_dim=pretrain_args.feature_dim).cuda()

    # Add special tokens for task training
    encoder_tokenizer.add_tokens(list(encoder_special_tokens.values()))
    decoder_tokenizer.add_tokens(list(decoder_special_tokens.values()))

    # Save modified tokenizers
    os.makedirs(os.path.join(pretrain_args.model_dir, "encoder", "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(pretrain_args.model_dir, "decoder", "tokenizer"), exist_ok=True)
    encoder_tokenizer.save_pretrained(os.path.join(pretrain_args.model_dir, "encoder", "tokenizer"))
    decoder_tokenizer.save_pretrained(os.path.join(pretrain_args.model_dir,"decoder", "tokenizer"))

    #Resize embeddings and update vocab sizes in configurations
    encoder.resize_token_embeddings(len(encoder_tokenizer))
    decoder.resize_token_embeddings(len(decoder_tokenizer))
    encoder_config.vocab_size = len(encoder_tokenizer)
    decoder_config.vocab_size = len(decoder_tokenizer)

    # Save updated configurations
    encoder_config.to_json_file(os.path.join(pretrain_args.model_dir, "encoder", CONFIG_NAME))
    decoder_config.to_json_file(os.path.join(pretrain_args.model_dir, "decoder", CONFIG_NAME))

    training_data = PretrainDataset(encoder_tokenizer, decoder_tokenizer, decoder_special_tokens,
                                    data_path=pretrain_args.training_data_path, data_type="train")

    max_train_batch_size = max(len(training_data) // pretrain_args.min_n_steps, pretrain_args.min_batch_size)
    train_loader = create_dataloader(training_data, batch_size=pretrain_args.batch_size, shuffle=True, enc_tokens=encoder_special_tokens,
                                     dec_tokens=decoder_special_tokens, encoder_tokenizer=encoder_tokenizer,
                                     decoder_tokenizer=decoder_tokenizer,max_batch_size=max_train_batch_size)

    params_to_optimize = list(encoder.named_parameters()) + list(decoder.named_parameters())
    disable_weight_decay = ["bias","LayerNorm.bias", "LayerNorm.weight"]
    grouped_params = [{"params":[p for n,p in params_to_optimize if not any(nd in n for nd in disable_weight_decay)],"weight_decay":pretrain_args.weight_decay},
                      {"params":[p for n,p in params_to_optimize if any(nd in n for nd in disable_weight_decay)],"weight_decay":0.0}]
    optimizer = AdamW(grouped_params, lr=pretrain_args.lr, eps=pretrain_args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_func = CrossEntropyLoss(ignore_index=-1)

    encoder.train()
    decoder.train()
    for epoch in range(pretrain_args.epochs):
        total_enc_loss, total_dec_loss, total_num_inputs = 0, 0, 0
        training_step = 0
        for encoder_inputs, decoder_inputs, decoder_labels in tqdm(train_loader, desc="Pretraining"):
            training_step += 1 

            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_labels = decoder_labels.cuda()

            # Encoder pass
            encoder_out = encoder(encoder_inputs)
            decoder_out = decoder(decoder_inputs, feature=encoder_out[0])
            encoder_logits = decoder_out[0]
            enc_loss = loss_func(encoder_logits.permute(0, 2, 1), decoder_labels)

            #Decoder Loss
            decoder_out = decoder(decoder_inputs)
            lm_logits = decoder_out[0]
            dec_loss = loss_func(lm_logits.permute(0, 2, 1), decoder_labels)
            loss = torch.mean(enc_loss) + torch.mean(dec_loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Track progress
            num_inputs = sum(len(x) for x in encoder_inputs)
            total_enc_loss += enc_loss.item()*num_inputs
            total_dec_loss += dec_loss.item()*num_inputs
            total_num_inputs += sum(len(x) for x in encoder_inputs)

            if training_step % 20000 == 0:
                torch.save(encoder.state_dict(),os.path.join(args.model_dir,"pretraining_encoder_" + str(epoch + 1)))
                torch.save(decoder.state_dict(),os.path.join(args.model_dir,"pretraining_decoder_" + str(epoch + 1)))

        print(f"Epoch {epoch}: Enc. Loss = {total_enc_loss/total_num_inputs}, Dec. Loss = {total_dec_loss/total_num_inputs}")

        # Save models
        torch.save(encoder.state_dict(),os.path.join(args.model_dir,"pretraining_encoder_" + str(epoch + 1)))
        torch.save(decoder.state_dict(),os.path.join(args.model_dir,"pretraining_decoder_" + str(epoch + 1)))
        

    #Save final models after training
    torch.save(encoder.state_dict(),os.path.join(args.model_dir,"pretrained_encoder"))
    torch.save(decoder.state_dict(),os.path.join(args.model_dir,"pretrained_decoder"))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--feature_dim", type=int, default=768)

    # Dataset Args
    parser.add_argument("--training_data_path", type=str, default="wiki40b/tokenized_train_data.pkl")

    # Training Args
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_batch_size", type=int, default=1)
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--min_n_steps", type=int, default=50000)
    args = parser.parse_args()

    model_name = f"bert-base_gpt2_{args.feature_dim}"
    model_dir = os.path.join(args.model_dir, model_name)
    args.model_dir = model_dir
    os.makedirs(model_dir, exist_ok=True)

    pretrain_model(args)

