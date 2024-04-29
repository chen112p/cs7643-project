import torch
import time
from metrics import eval


class SolverLLM():
    def __init__(self, model, optimizer, criterion, model_type = "LLM"):
        self.epoch = None
        self.model_type = model_type
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, data_loader):
        iter_time = eval.AverageMeter()
        losses = eval.AverageMeter()
        acc = eval.AverageMeter()

        cm = torch.zeros(self.model.num_labels, self.model.num_labels)

        for idx, data in enumerate(data_loader):
            start = time.time()
            if self.model_type == "RNN":
                x = data[0]
                y = data[1]
                out = self.model.forward(x)
                loss = self.criterion(out, y)
            elif self.model_type == "LLM":
                x = data['input_ids'], data['attention_mask']
                y = data['labels']
                out = self.model.forward(x)
                loss = self.criterion(out, y)
            elif self.model_type == "QLORA":
                y = data['labels']
                outputs = self.model.forward(**data)
                loss = outputs.loss
                out = outputs.logits
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_acc = eval.accuracy(out, y)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                        'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t').format(
                            self.epoch,
                            idx,
                            len(data_loader),
                            iter_time=iter_time,
                            loss=losses,
                            top1=acc))
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Train Accuracy of Class {}: {:.4f}".format(i, acc_i))
        self.train_class0_acc = per_cls_acc[0]
        self.train_class1_acc = per_cls_acc[1]

        print("* Train Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm, losses.avg, iter_time.avg

    def validate(self, val_loader):
        iter_time = eval.AverageMeter()
        losses = eval.AverageMeter()
        acc = eval.AverageMeter()

        cm = torch.zeros(self.model.num_labels, self.model.num_labels)

        # evaluation loop
        for idx, data in enumerate(val_loader):
            start = time.time()

            with torch.no_grad():
                if self.model_type == "RNN":
                    x = data[0]
                    y = data[1]
                    out = self.model.forward(x)
                    loss = self.criterion(out, y)
                elif self.model_type == "LLM":
                    x = data['input_ids'], data['attention_mask']
                    y = data['labels']
                    out = self.model.forward(x)
                    loss = self.criterion(out, y)
                elif self.model_type == "QLORA":
                    y = data['labels']
                    outputs = self.model.forward(**data)
                    loss = outputs.loss
                    out = outputs.logits

            batch_acc = eval.accuracy(out, y)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t').format(
                        self.epoch,
                        idx,
                        len(val_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc))
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Valid Accuracy of Class {}: {:.4f}".format(i, acc_i))
        self.val_class0_acc = per_cls_acc[0]
        self.val_class1_acc = per_cls_acc[1]
        print("* Valid Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm, losses.avg, iter_time.sum