"""
Base RNN model to classify Hateful Tweets
"""

import torch


class MyModel(torch.nn.Module):
    def __init__(self, alphabet, longest_sent, embedding_size, hidden_size, num_classes, device):
        super(MyModel, self).__init__()
        self.alphabet = alphabet
        self.longest_sent = longest_sent
        self.embedding_size = embedding_size # some lower level dimensions
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        # Initialize layers
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.alphabet, # alphabet size from embedding
            embedding_dim=self.embedding_size, # some lower level dimensions
            padding_idx=None, 
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            _freeze=False,
            device=self.device, 
            dtype=None
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_size, #from embedding
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
            proj_size=0,
            device=self.device,
            dtype=None
        )
        self.linear = torch.nn.Linear(
            in_features = self.hidden_size*self.longest_sent,
            out_features=self.num_classes,
            bias=True,
            device=self.device,
            dtype=None
        )

    def forward(self, x):
        outs = None
        # print("x",x.shape)
        x = x.to(torch.int64)
        x = self.embedding(x)
        # print("embedding", x.shape)
        x, (h, c) = self.lstm(x)
        # print("lstm", x.shape) 
        x = torch.flatten(x, 1, -1)
        # print("flatten", x.shape)
        x = self.linear(x)
        # print("linear", x.shape)
        outs = x.clone()
        return outs
        