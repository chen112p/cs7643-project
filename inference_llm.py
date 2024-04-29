import torch
from data import dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import argparse

tokenizer_dict = {
    'distilroberta_classifier': 'roberta-base',
    'roberta_classifier': 'roberta-base',
    'distilroberta_lora_classifier': 'roberta-base',
    'distilroberta_qlora_classifier': 'roberta-base',
    'roberta_lora_classifier': 'roberta-base',
    'roberta_qlora_classifier': 'roberta-base',
}

def main(modelname,
        device):
    model_state = torch.load(r'saved_models/{}'.format(modelname))
    if "roberta" in modelname.lower() and "lora" in modelname.lower():
        from models import roberta_lora_classifier as rlc
        model = rlc.RobertaLoraClassifier(dropout_rate = 0.5,lora_rank = model_state['lora_rank'])
    elif "roberta" in modelname.lower() and "lora" not in modelname.lower():
        from models import roberta_classifier as rc
        model = rc.RobertaClassifier(dropout_rate = 0.5)
    model.load_state_dict(model_state, strict=False)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    test_dataset = dataset.RoBerta_Dataset('data/test.tsv',
                                                    tokenizer,
                                                    device = 'cuda')
    test_loader = DataLoader(test_dataset,
                            batch_size = 16,
                            shuffle=False)

    y_pred = np.array([])
    y_true = np.array([])
    with torch.no_grad():
      for idx, data in enumerate(test_loader):
          x = data['input_ids'], data['attention_mask']
          y = data['labels'].cpu().numpy()
          y_true = np.concatenate((y_true, y))
          y_pred = np.concatenate([y_pred,torch.argmax(F.softmax(model(x),dim = 1).cpu(),dim=1)])
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    col_label = ['Predict Negative', 'Predict Positive']
    row_label = ['True Negative', 'True Positive']
    df_conf_matrix = pd.DataFrame(conf_matrix, index=row_label, columns=col_label)
    df_conf_matrix.to_csv(r'figs/{}.csv'.format(modelname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", dest="device", required=True,
                          help="type of device")
    parser.add_argument("-modelname", dest="modelname", required=True,
                          help="model_name")
    args = parser.parse_args()

    main(modelname = args.modelname,
          device=args.device)