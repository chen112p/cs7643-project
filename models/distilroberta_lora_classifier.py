import torch
from transformers import AutoModel
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention



class RobertaLoraClassifier(torch.nn.Module):
    def __init__(self,
                dropout_rate):
        super().__init__()
        print('asdf')
        self.num_class = 2
        self.pretrain_model = AutoModel.from_pretrained('distilroberta-base', return_dict = True)
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
        print(model.named_childred())
        """
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
        """