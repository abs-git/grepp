import torch


def get_metric(TP, FP, TN, FN):
    TP = float(TP)
    FP = float(FP)
    TN = float(TN)
    FN = float(FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return precision, recall, f1, miss_rate, accuracy

def outcome(y, y_pred):
    preds = torch.argmax(y_pred, dim=1)

    correct = (preds == y)
    incorrect = ~correct

    tp = correct.sum().item()
    fp = incorrect.sum().item()
    fn = fp
    tn = 0

    return tp, fp, tn, fn

if __name__=='__main__':

    actuals = [torch.randn(4,3,360), torch.randn(4,3,640)]
    preds = [torch.randn(4,3,360), torch.randn(4,3,640)]

    TP, FP, FN = outcome(actuals, preds, rank=1)
