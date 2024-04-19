from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModel
import torch.nn.functional as F
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
import math

class LoraRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, r=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.all_head_size
        self.lora_query_matrix_B = nn.Parameter(torch.zeros(d, r))
        self.lora_query_matrix_A = nn.Parameter(torch.randn(r, d))
        self.lora_value_matrix_B = nn.Parameter(torch.zeros(d, r))
        self.lora_value_matrix_A = nn.Parameter(torch.randn(r, d))
    def lora_query(self, x):
        lora_full_query_weights = torch.matmul(self.lora_query_matrix_B, self.lora_query_matrix_A)
        return self.query(x) + F.linear(x, lora_full_query_weights)
    def lora_value(self,x):
        lora_full_value_weights = torch.matmul(self.lora_value_matrix_B, self.lora_value_matrix_A)
        return self.value(x) + F.linear(x, lora_full_value_weights)
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
        """Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
        but replaced the query and value calls with calls to the lora_query and lora_value functions.
        """
        mixed_query_layer = self.lora_query(hidden_states) # lora query 

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(encoder_hidden_states)) # lora value
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states)) # lora value
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    


class RobertaLoraClassifier(torch.nn.Module):
    def __init__(self,
                dropout_rate):
        super().__init__()
        self.num_class = 2
        self.pretrain_model = AutoModel.from_pretrained('roberta-base', return_dict = True)
        self.hidden = torch.nn.Linear(self.pretrain_model.config.hidden_size, 
                                self.pretrain_model.config.hidden_size)
        self.classifier = torch.nn.Linear(self.pretrain_model.config.hidden_size,
                                        self.num_class)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.model_config = self.pretrain_model.config
        # lora
        self.replace_multihead_attention()
        self.freeze_parameters_except_lora_and_bias()


    def forward(self, x):
        (input_ids, attention_mask) = x
        # roberta model
        output = self.pretrain_model(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # neural network classification layer
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(pooled_output)
        logits = self.classifier(pooled_output)        
        return(logits)
    def replace_multihead_attention(self):
        self.nr_replaced_modules = 0
        self.replace_multihead_attention_recursion(self.pretrain_model)

    def replace_multihead_attention_recursion(self, model):
        for name, module in model.named_children():
            if isinstance(module, RobertaSelfAttention):
                self.nr_replaced_modules += 1

                # Create a new LoraMultiheadAttention layer
                new_layer = LoraRobertaSelfAttention(r=self.lora_rank, config=self.model_config)

                # Get the state of the original layer
                state_dict_old = module.state_dict()

                # Load the state dict to the new layer
                new_layer.load_state_dict(state_dict_old, strict=False)

                # Get the state of the new layer
                state_dict_new = new_layer.state_dict()

                # Compare keys of both state dicts
                keys_old = set(state_dict_old.keys())
                keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
                assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring lora parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. LoRA): {keys_new}"

                # Replace the original layer with the new layer
                setattr(model, name, new_layer)

            else:
                # Recurse on the child modules
                self.replace_multihead_attention_recursion(module)
        
    def freeze_parameters_except_lora_and_bias(self):
        """
        Freezes all parameters in the model, except those in LoRA layers, the finetune head, and bias parameters, if specified.
        All lora parameters are identified by having a name that starts with *lora_*.
        All finetune head parameters are identified by having a name that starts with *finetune_head_*.
        """
        for name, param in self.pretrain_model.named_parameters():
            if ("lora_" in name) or ("finetune_head_" in name) or (self.train_biases and "bias" in name) \
                or (self.train_embeddings and "embeddings" in name) or (self.train_layer_norms and "LayerNorm" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False