from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


class UCC_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 attributes,
                 max_token_length:int = 128,
                sample = 5000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_len = max_token_length
        self.sample = sample
        self._prepare_data()
    def _prepare_data(self):
        data = pd.read_csv(self.data_path)
        data['unhealthy'] = np.where(data['healthy']==1, 0, 1)
        if self.sample is not None:
            unhealthy = data.loc[data[self.attributes].sum(axis=1) > 0]
            #print(data[attributes].shape)
            healthy = data.loc[data[self.attributes].sum(axis=1) == 0]
            #print(healthy.shape)
            self.data = pd.concat([unhealthy,
                                   healthy.sample(self.sample, random_state = 7)])
        else:
            self.data = data
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, index):
        item = self.data.iloc[index]
        comment = str(item.comment)
        attributes = torch.FloatTensor(item[self.attributes].to_numpy().astype(np.int16))
        tokens = self.tokenizer.encode_plus(comment, 
                                            add_special_tokens=True,
                                            return_tensors = 'pt',
                                            truncation = True,
                                            max_length = self.max_token_len,
                                            padding = 'max_length',
                                           return_attention_mask=True)
        return({'input_ids': tokens.input_ids.flatten(), 
                'attention_mask': tokens.attention_mask.flatten(),
               'labels': attributes})

  
class UCC_Data_Module:
    def __init__(self, train_path, val_path, attributes, batch_size=16, max_token_length=128, model_name='roberta-base'):
        self.train_path = train_path
        self.val_path = val_path
        self.attributes = attributes
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = UCC_Dataset(self.train_path, self.tokenizer, self.attributes)
            self.val_dataset = UCC_Dataset(self.val_path, self.tokenizer, self.attributes)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

