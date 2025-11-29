import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        return F.cross_entropy(pred, target, weight=self.weight)

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        """
        probas: [B, C, N]
        labels: [B, N]
        """
        probas = F.softmax(probas, dim=1)
        probas = probas.permute(0, 2, 1).contiguous().view(-1, probas.size(1))
        labels = labels.view(-1)
        
        if self.ignore is not None:
            valid = (labels != self.ignore)
            probas = probas[valid]
            labels = labels[valid]
            
        return lovasz_softmax_flat(probas, labels, classes=self.classes)

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(CombinedLoss, self).__init__()
        self.ce = WeightedCrossEntropyLoss(weight=weight)
        self.lovasz = LovaszSoftmaxLoss(ignore=ignore_index)

    def forward(self, pred, target):
        # pred: [B, C, N]
        # target: [B, N]
        
        # CE expects [B, C, N], target [B, N]
        loss_ce = self.ce(pred, target)
        
        # Lovasz expects [B, C, N], target [B, N] (handled inside)
        loss_lovasz = self.lovasz(pred, target)
        
        return 0.7 * loss_ce + 0.3 * loss_lovasz
