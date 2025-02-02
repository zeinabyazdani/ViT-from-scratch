
import torch

class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def accuracy(label, label_pred, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """

    accuracy = 0
    with torch.no_grad():
        label_pred = torch.argmax(label_pred, dim=1)
        label = torch.argmax(label, dim=1)

        equals = label_pred == label
        accuracy += torch.sum(equals.type(torch.FloatTensor))
    accuracy /= label.size(0)
    return accuracy
