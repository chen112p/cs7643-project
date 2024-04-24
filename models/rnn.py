"""
Base RNN model to classify Hateful Tweets
"""

from data.dataset import get_data, convert_tweet2tensor
import os
import pandas as pd
import torch


class MyModel(torch.nn.Module):
    def __init__(self, alphabet, longest_sent, embedding_size, hidden_size, num_layers, dropout, num_labels, bidirectional, device):
        super(MyModel, self).__init__()
        self.alphabet = alphabet
        self.longest_sent = longest_sent
        self.num_labels = num_labels
        self.embedding_size = embedding_size # some lower level dimensions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
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
            bidirectional=self.bidirectional,
            proj_size=0,
            device=self.device,
        )
        # D if bidirectional LSTM
        D = 1
        if self.bidirectional:
            D = 2
        self.linear = torch.nn.Linear(
            in_features = self.hidden_size*self.longest_sent*D,
            out_features=self.num_labels,
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


def model_inference(
    model: torch.nn.Module,
    train_file: os.PathLike = None,
    test_file: os.PathLike = None,
    sample_size: int = 10,
):
    _, _, alphabet, longest_sent, _, _ = get_data(train_file, test_file)
    test_df = pd.read_csv(test_file, sep='\t',skiprows=0, encoding = 'utf-8')
    test_sample = test_df.sample(sample_size, random_state=0)
    for idx, row in test_sample.iterrows():
        test_tensor = convert_tweet2tensor(row['text'], alphabet, longest_sent)
        print(f"test text: {test_df.loc[idx,'text']}")
        pred_prob = model.forward(test_tensor)
        pred_label = torch.argmax(pred_prob)
        print(f"true label: {test_df.loc[idx,'HS']}")
        print(f"best model predicted label: {pred_label}")
        print()