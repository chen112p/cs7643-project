import torch
from transformers import AutoModel
import torch.nn.functional as F

class RobertaClassifier(torch.nn.Module):
    def __init__(self,
                dropout_rate):
        super().__init__()
        self.num_labels = 2
        self.pretrain_model = AutoModel.from_pretrained('roberta-base', return_dict = True)
        self.hidden = torch.nn.Linear(self.pretrain_model.config.hidden_size, 
                                self.pretrain_model.config.hidden_size)
        self.classifier = torch.nn.Linear(self.pretrain_model.config.hidden_size,
                                        self.num_labels)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = torch.nn.Dropout(dropout_rate)

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
