import math
import json
import torch

from torch import nn
from pytorch_transformers.modeling_utils import PreTrainedModel, PretrainedConfig, prune_linear_layer
from .model_utils import load_bert_weights

bert_weight_map = {"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"}
bert_config_map = {"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"}

class BertConfig(PretrainedConfig):

    pretrained_config_archive_map = bert_config_map

    def __init__(
        self,
        vocab_size_or_config_json_file=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                " or the path to a pretrained model config file (str)"
            )

class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = bert_weight_map
    load_tf_weights = load_bert_weights
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.output_attentions = True

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_atten = BertSelfAttention(config)

        # Output layer
        self.atten_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.atten_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.atten_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self_atten.num_attention_heads, self.self_atten.attention_head_size)
        # Convert to set and emove already pruned heads
        heads = set(heads) - self.pruned_heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self_atten.query = prune_linear_layer(self.self_atten.query, index)
        self.self_atten.key = prune_linear_layer(self.self_atten.key, index)
        self.self_atten.value = prune_linear_layer(self.self_atten.value, index)
        self.atten_dense = prune_linear_layer(self.atten_dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self_atten.num_attention_heads = self.self_atten.num_attention_heads - len(heads)
        self.self_atten.all_head_size = (self.self_atten.attention_head_size * self.self_atten.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self_atten(input_tensor, attention_mask, head_mask)

        # Applying output layer
        hidden_state = self.atten_dropout(self.atten_dense(self_outputs[0]))
        attention_output = self.atten_norm(hidden_state + input_tensor)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs

def bert_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)

        # Intermediate Layer
        self.dense_intermed = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_intermed = bert_gelu

        # Output Layer
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None):

        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # Intermediate Layer
        intermediate_output = self.act_intermed(self.dense_intermed(attention_output))

        # Output layer
        hidden_out = self.dense_out(intermediate_output)
        hidden_out = self.dropout_out(hidden_out)
        layer_output = self.norm_out(hidden_out+attention_output)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.out_attention = True
        self.out_hidden_state = True
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_state, attention_mask, head_mask = None):
        all_hidden, all_attention = (), ()
        for i, layer_module in enumerate(self.layer):
            if self.out_hidden_state:
                all_hidden = all_hidden + (hidden_state, )
            layer_out = layer_module(hidden_state, attention_mask, head_mask[i])
            hidden_state = layer_out[0]

            if self.out_attention:
                all_attention = all_attention + (layer_out[1],)
        if self.out_hidden_state:
            all_hidden = all_hidden + (hidden_state, )
        output = (hidden_state, )
        if self.out_hidden_state:
            output = output + (all_hidden,)
        if self.out_attention:
            output = output + (all_attention,)
        return output

class BertEmbeddings(nn.Module):

    def __init__(self, config):

        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, token_type_ids=None, position_ids=None):

        if position_ids is None:
            position_ids = torch.arange(x.size(1),dtype=torch.long,device=x.device).unsqueeze(0).expand_as(x)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)
        
        word_embedding = self.word_embeddings(x)
        pos_embedding = self.position_embeddings(position_ids)
        token_type_embedding = self.token_type_embedding(token_type_ids)

        embeddings = word_embedding + pos_embedding + token_type_embedding
        return self.dropout(self.norm(embeddings))

class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Pooling
        self.pool_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pool_activation = nn.Tanh()

        self.init_weights()

    def _resize_token_embeddings(self, new_token_size):
        resized_embedding = self._get_resized_embeddings(self.embeddings.word_embeddings, new_token_size)
        self.embeddings.word_embeddings = resized_embedding
        return self.embeddings.word_embeddings

    def forward(self, x, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):

        # Initialize masks if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)

        # 3D attention mask for simplified causal masking
        dtype = next(self.parameters()).dtype
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype)

        if head_mask is None: # No mask provided
            head_mask = [None for _ in range(self.config.num_hidden_layers)]
        else:
            if head_mask.dim() == 1: # Mask has shape [num_heads]
                head_mask = (head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1,-1,-1,-1)
            elif head_mask.dim() == 2: # Mask has shape [num_hidden_layers, num_heads]
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        embedded = self.embeddings(x, position_ids=position_ids, token_type_ids=token_type_ids)
        encoded, hidden_states, attention = self.encoder(embedded, attention_mask, head_mask=head_mask)
        pooled = self.pool_activation(self.pool_dense(encoded))

        return (encoded, pooled, hidden_states, attention)

class Encoder(nn.Module):
    def __init__(self, transformer, tokenizer, feature_dim, activation=torch.tanh):
        super().__init__()

        self.feature_dim = feature_dim

        self.transformer = transformer
        self.connector_layer = nn.Linear(self.transformer.config.hidden_size, self.feature_dim, bias=False)

        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.config = transformer.config

    def forward(self, input_ids, attention_mask=None):

        if attention_mask is None:
            attention_mask = input_ids.eq(self.pad_token_id).logical_not().float()

        last_hidden_state, cls_pooled_hidden_state = self.transformer(input_ids, attention_mask=attention_mask)[:2]
        feature = self.connector_layer(cls_pooled_hidden_state)

        return feature, last_hidden_state

    def resize_token_embeddings(self, len_tokenizer):
        self.transformer.resize_token_embeddings(len_tokenizer)


