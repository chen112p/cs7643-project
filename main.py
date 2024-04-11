import torch
from data import dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from models import distilroberta_classifier as drc
from solver import solver_llm
import copy
import matplotlib.pyplot as plt
from datetime import datetime
timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
tokenizer_dict = {
    'distilroberta_classifer': 'roberta-base'
}
#hard code some input parameters
max_epoch = 10
train_batch_size = 256
test_batch_size = 256
device = 'cuda'
train_file="data/train_en.tsv"
test_file="data/dev_en.tsv"
model_name = 'distilroberta_classifer'
lr = 1e-5
dropout_rate = 0.5

def main():  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[model_name])
    

    train_dataset = dataset.RoBerta_Dataset(train_file,tokenizer,device = device)
    test_dataset = dataset.RoBerta_Dataset(test_file,tokenizer,device = device)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    model = drc.RobertaClassifier(dropout_rate = dropout_rate)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr = lr, params=model.parameters())

    solver = solver_llm.SolverLLM(model,
                                optimizer,
                                criterion)
    best = 0.0
    best_cm = None
    best_model = None
    train_acc_epoch = []
    valid_acc_epoch = []
    for epoch in range(max_epoch):
        solver.epoch = epoch
        # train loop
        train_acc, train_cm = solver.train(train_loader)
        train_acc_epoch.append(train_acc.detach().cpu().numpy())

        # validation loop
        valid_acc, valid_cm = solver.validate(test_loader)
        valid_acc_epoch.append(valid_acc.detach().cpu().numpy())

        if valid_acc > best:
            best = valid_acc
            best_cm = valid_cm
            best_model = copy.deepcopy(model)
    
    torch.save(best_model.state_dict(), 
              'models/{}_{}'.format(model_name, timestamp_str))
    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    plt.figure()
    plt.plot(range(max_epoch), train_acc_epoch, label='train')
    plt.plot(range(max_epoch), valid_acc_epoch, label='validation')
    plt.legend()
    plt.title("accuracy curve")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('figs/{}_{}.png'.format(model_name, timestamp_str))

if __name__ == '__main__':
    main()