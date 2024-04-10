import torch
import time
from metrics import eval

class SolverLLM():
    def __init__(self,
                 model,
                 optimizer,
                 criterion):
        self.epoch = None
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    def train(self, data_loader):
        iter_time = eval.AverageMeter()
        losses = eval.AverageMeter()
        acc = eval.AverageMeter()

        cm = torch.zeros(self.model.num_class, self.model.num_class)

        for idx, data in enumerate(data_loader):
            start = time.time()
            out = self.model.forward(data['input_ids'], data['attention_mask'])
            
            loss = self.criterion(out, data['labels'])
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_acc = eval.accuracy(out, data['labels'])

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(data['labels'].view(-1), preds.view(-1)):
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

        print("* Train Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm

    def validate(self,epoch, val_loader, model, criterion):
        iter_time = eval.AverageMeter()
        losses = eval.AverageMeter()
        acc = eval.AverageMeter()

        cm = torch.zeros(model.num_class, model.num_class)
        # evaluation loop
        for idx, (data, target) in enumerate(val_loader):
            start = time.time()

            with torch.no_grad():
                out = model.forward(data)
                loss = criterion(out, target)
                
            batch_acc = eval.accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t').format(
                        epoch,
                        idx,
                        len(val_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc))
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Valid Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("* Valid Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm