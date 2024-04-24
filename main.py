import os
import torch
from data import dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from solver import solver_llm
import copy
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
import argparse
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)


timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
tokenizer_dict = {
    'distilroberta_classifier': 'roberta-base',
    'roberta_classifier': 'roberta-base',
    'distilroberta_lora_classifier': 'roberta-base',
    'distilroberta_qlora_classifier': 'roberta-base',
}

def main(config_file): 
    device = config_file['device']

    if config_file['model_name'] == "RNN":
        train_dataset, test_dataset, alphabet, longest_sent, _, _ = dataset.get_data(config_file['train_file'], config_file['test_file'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[config_file['model_name']])
        train_dataset = dataset.RoBerta_Dataset(config_file['train_file'],
                                                tokenizer,
                                                device = device)
        test_dataset = dataset.RoBerta_Dataset(config_file['test_file'],
                                                tokenizer,
                                                device = device)
        
    train_loader = DataLoader(train_dataset, 
                            batch_size=config_file['train_batch_size'], 
                            shuffle=True)
    test_loader = DataLoader(test_dataset, 
                            batch_size = config_file['test_batch_size'],
                            shuffle=True)
                            
    if config_file['model_name'] == 'distilroberta_classifier':
        from models import distilroberta_classifier as drc
        model = drc.RobertaClassifier(dropout_rate = config_file['dropout_rate'])
        model_type = "LLM"
        optimizer = torch.optim.Adam(lr = config_file['lr'], params=model.parameters())
    elif config_file['model_name'] == 'roberta_classifier':
        from models import roberta_classifier as rc
        model = rc.RobertaClassifier(dropout_rate = config_file['dropout_rate'])
        model_type = "LLM"
        optimizer = torch.optim.Adam(lr = config_file['lr'], params=model.parameters())
    elif config_file['model_name'] == 'distilroberta_lora_classifier':
        from models import distilroberta_lora_classifier as drlc
        model = drlc.RobertaLoraClassifier(dropout_rate = config_file['dropout_rate'])
        model_type = "LLM"
        optimizer = torch.optim.Adam(lr = config_file['lr'], params=model.parameters())
    elif config_file['model_name'] == 'RNN':
        from models import rnn
        model = rnn.MyModel(
            len(alphabet),
            longest_sent,
            config_file['embedding_size'],
            config_file['hidden_size'],
            config_file['num_layers'],
            config_file['dropout_rate'],
            config_file['num_labels'],
            config_file['bidrectional'],
            device)
        model_type = "RNN"
        optimizer = torch.optim.Adam(lr = config_file['lr'], params=model.parameters())
    elif config_file['model_name'] == 'distilroberta_qlora_classifier':
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        # Configuration to load a quantized model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit loading
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base',
            torch_dtype="auto", quantization_config=bnb_config)
        # Config for the LoRA Injection via PEFT
        peft_config = LoraConfig(
            r=config_file['qlora_rank'], # rank dimension of the LoRA injected matrices
            lora_alpha=config_file['qlora_alpha'], # parameter for scaling, use 8 here to make it comparable with our own implementation
            target_modules=config_file['qlora_target_modules'], # be precise about dense because classifier has dense too
            # modules_to_save=["LayerNorm", "classifier", "qa_outputs"], # Retrain the layer norm; classifier is the fine-tune head; qa_outputs is for SQuAD
            lora_dropout=config_file['dropout_rate'], # dropout probability for layers
            bias=config_file['qlora_bias'], # none, all=retrain all biases of all modules, or lora_only
        )
        # Prepare the model for quantization
        model = get_peft_model(model, peft_config)
        model_type = "QLORA"
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr = config_file['lr'])
    
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    

    solver = solver_llm.SolverLLM(model,
                                optimizer,
                                criterion,
                                model_type)
    best = 0.0
    best_cm = None
    best_model = None
    train_acc_epoch = []
    train_loss_epoch = []
    valid_acc_epoch = []
    valid_loss_epoch = []
    for epoch in range(config_file['max_epoch']):
        solver.epoch = epoch
        # train loop
        train_acc, train_cm, train_loss = solver.train(train_loader)
        train_acc_epoch.append(train_acc.detach().cpu().numpy())
        train_loss_epoch.append(train_loss)

        # validation loop
        valid_acc, valid_cm, valid_loss = solver.validate(test_loader)
        valid_acc_epoch.append(valid_acc.detach().cpu().numpy())
        valid_loss_epoch.append(valid_loss)

        if valid_acc > best:
            best = valid_acc
            best_cm = valid_cm
            best_model = copy.deepcopy(model)
        break
    if config_file['model_name'] != 'distilroberta_lora_classifier':
        torch.save(best_model.state_dict(), 
                  'models/{}_{}'.format(config_file['model_name'], timestamp_str))
    else:
        print(best_model.lora_query_matrix_B.shape)
    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    f,ax = plt.subplots(2,1)
    ax[0].plot(range(config_file['max_epoch']), train_loss_epoch, label='train')
    ax[0].plot(range(config_file['max_epoch']), valid_loss_epoch, label='validation')
    ax[0].set_title('loss curve')
    ax[0].set_xlabel('epoch')
    ax[0].set_label('loss')
    ax[1].plot(range(config_file['max_epoch']), train_acc_epoch, label='train')
    ax[1].plot(range(config_file['max_epoch']), valid_acc_epoch, label='validation')
    ax[1].legend()
    ax[1].set_title("accuracy curve")
    ax[1].set_xlabel('epoch')
    ax[1].set_label('accuracy')
    f.tight_layout()
    os.makedirs('figs',exist_ok=True)
    f.savefig('figs/{}_{}.png'.format(config_file['model_name'], timestamp_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", dest="config_file", required=True,
                          help="Path to the configuration file")
    args = parser.parse_args()
    config_file = read_yaml(os.path.join('config',args.config_file))
    main(config_file)