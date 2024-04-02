"""
Utility to process raw tweets
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from typing import Union

import emoji
import re
import unicodedata


def get_data(
    train_file: Union[str, os.PathLike] = None,
    test_file: Union[str, os.PathLike] = None,
):
    
    assert train_file is not None, "train_file location cannot be found"
    assert test_file is not None, "test_file location cannot be found"

    # train tweet and hateful labels
    train_df = pd.read_csv(train_file, sep='\t',skiprows=0, encoding = 'utf-8')
    train_df['clean text'] = train_df['text'].apply(give_emoji_free_text)
    list_of_sentences_train = train_df['clean text'].values.tolist()
    print(f"first 3 raw tweets: {train_df['text'].values.tolist()[:3]}")
    print(f"first 3 processed tweets: {list_of_sentences_train[:3]}")
    train_y = torch.from_numpy(train_df['HS'].values)
    # test tweet and hateful labels
    test_df = pd.read_csv(test_file, sep='\t',skiprows=0, encoding = 'utf-8')
    test_df['clean text'] = test_df['text'].apply(give_emoji_free_text)
    list_of_sentences_test = test_df['clean text'].values.tolist()
    test_y = torch.from_numpy(test_df['HS'].values)

    train_max_length = list()
    for x in list_of_sentences_train:
        train_max_length.append(len(x))
    train_longest_sent = max(train_max_length)

    test_max_length = list()
    for x in list_of_sentences_test:
        test_max_length.append(len(x))
    test_longest_sent = max(test_max_length)

    if train_longest_sent >= test_longest_sent:
        longest_sent = train_longest_sent
        print(f"using train_longest_sent: {train_longest_sent}")
    else:
        longest_sent = test_longest_sent
        print(f"using test_longest_sent: {test_longest_sent}")
    
    alphabet = dict()
    train_words = []
    for sent in list_of_sentences_train:
        words_tensor = []
        words = sent.split()
        for w in words:
            if w in alphabet:
                words_tensor.append(alphabet[w])
            else:
                alphabet[w] = len(alphabet)
                words_tensor.append(alphabet[w])
        padded_words = torch.nn.functional.pad(torch.Tensor(words_tensor), (0, longest_sent - len(words_tensor)))
        train_words.append(torch.Tensor(padded_words))
    train_x = torch.stack(train_words, dim=0) #pad_sequence(train_words, batch_first=True, padding_value=0)
    
    print("train_x shape", train_x.shape)
    print("train_words", len(train_words))
    print("len alphabet", len(alphabet))

    # Create a TensorDataset
    train_dataset = TensorDataset(train_x, train_y)

    test_words = []
    for sent in list_of_sentences_test:
        words_tensor = []
        words = sent.split()
        for w in words:
            if w in alphabet:
                words_tensor.append(alphabet[w])
            else:
                alphabet[w] = len(alphabet)
                words_tensor.append(alphabet[w])
        padded_words = torch.nn.functional.pad(torch.Tensor(words_tensor), (0, longest_sent - len(words_tensor)))
        test_words.append(padded_words)
    test_x = torch.stack(test_words, dim=0)
    print("test_x shape", test_x.shape)
    print("test_words", len(test_words))

    # Create a TensorDataset
    test_dataset = TensorDataset(test_x, test_y)

    return train_dataset, test_dataset, alphabet, longest_sent, train_x, test_x


def unicodeToAscii(s: str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s: str):
    s = unicodeToAscii(s.lower().strip())
    s = ''.join([str for str in s.split() if not any([i for i in str if i in '@#$/:'])])
    s = re.sub("[^a-zA-Z]+", "", s)
    return s.strip()

def give_emoji_free_text(text):
    # allchars = [str for str in text]
    # emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    # new_text = re.sub(emoji.get_emoji_regexp(), "", text).lower()
    # ALPHABET = 'abcdefhijklmnopqrstuvwxyz#'
    # clean_text = ' '.join([str for str in new_text.lower().split() if all([i if i in ALPHABET else False for i in str])])
    # clean_text = ' '.join([str for str in text.lower().split() if not any([i if i in '@#'  else False for i in str])])
    # print("original", text)
    # print("clean_text",clean_text)
    list1 = [normalizeString(s) for s in text.split()]

    # remove empty strings
    list2 = [s for s in list1 if s]

    clean_text = ' '.join([s for s in list2])
        
    return clean_text