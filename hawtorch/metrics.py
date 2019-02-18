import torch
import numpy as np

class ClassificationAverager:
    """ statistics for classification """
    def __init__(self, nCls, eps=1e-5):
        self.nCls = nCls
        self.eps = eps
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)

    def clear(self):
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)

    def update(self, outputs, truths):
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy() # [B, N]
        labels = truths.detach().cpu().numpy() # [B, ]
        batch_size = labels.shape[0]
        self.N += batch_size
        for Cls in range(self.nCls):
            true_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels == Cls))
            true_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels != Cls))
            false_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels != Cls))
            false_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels == Cls))
            self.table[Cls] += [true_positive, true_negative, false_positive, false_negative]
    
    def report(self, multiclass=True):
        precisions = []
        recalls = []
        for Cls in range(self.nCls):
            precision = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,3] + self.eps) # TP / (TP + FN)
            recall = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,2] + self.eps) # TP / (TP + FP)
            precisions.append(precision)
            recalls.append(recall)
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        accuracy = total_TP/self.N

        text = f"Accuracy = {accuracy:.4f}({total_TP}/{self.N})\n"
        if multiclass:
            for Cls in range(self.nCls):
                text += "\tClass {}: precision = {:.3f} recall = {:.3f}\n".format(Cls, precisions[Cls], recalls[Cls])

        return text