"""
Base RNN model to classify Hateful Tweets
"""

import torch


class MyModel(torch.nn.Module):
    def __init__(self, alphabet, longest_sent, embedding_size, hidden_size, num_layers, dropout, num_classes, bidrectional, device):
        super(MyModel, self).__init__()
        self.alphabet = alphabet
        self.longest_sent = longest_sent
        self.embedding_size = embedding_size # some lower level dimensions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.bidrectional = bidrectional
        self.device = device
        # Initialize layers
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.alphabet, # alphabet size from embedding
            embedding_dim=self.embedding_size, # some lower level dimensions
            device=self.device, 
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidrectional,
            proj_size=0,
            device=self.device,
        )
        # D if bidirectional LSTM
        D = 1
        if self.bidrectional:
            D = 2
        self.linear = torch.nn.Linear(
            in_features = self.hidden_size*self.longest_sent*D,
            out_features=self.num_classes,
            bias=False,
            device=self.device,
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
        