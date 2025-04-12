import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy_Loss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(CrossEntropy_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y, y_pred):
        probs = F.softmax(y_pred, dim=1) + self.epsilon
        targets_one_hot = F.one_hot(y, num_classes=y_pred.size(1)).float()
        loss = -torch.sum(targets_one_hot * torch.log(probs), dim=1)
        return torch.mean(loss)

class SoftmaxFocal_Loss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(SoftmaxFocal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y, y_pred):
        log_probs = F.log_softmax(y_pred, dim=1)
        probs = torch.exp(log_probs)

        y = y.view(-1, 1)
        log_p_t = log_probs.gather(1, y).squeeze(1)
        p_t = probs.gather(1, y).squeeze(1)

        focal_term = (1 - p_t) ** self.gamma
        loss = -self.alpha * focal_term * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DICE_Smooth_Loss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(DICE_Smooth_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y, y_pred):
        return 1. - ((2 * torch.sum( y_pred * y) + self.epsilon) / (torch.sum(y_pred) + torch.sum(y) + self.epsilon))


class BCE_Loss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(BCE_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y, y_pred):
        return - torch.mean(y * torch.log(y_pred + self.epsilon) + 
                           (1 - y) * torch.log(1 - y_pred + self.epsilon))


class WBCE_Loss(nn.Module):
    def __init__(self, pos_weight=5.0, epsilon=1e-12):
        super(WBCE_Loss, self).__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, y, y_pred):
        return -1 * (self.pos_weight * ((1 - y_pred) ** 2) * y * torch.log(torch.clamp(y_pred, min=self.epsilon, max=1))
                                     + (y_pred ** 2) * (1 - y) * torch.log(torch.clamp(1 - y_pred, min=self.epsilon, max=1))
                    ).sum()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)

        ce_loss = -y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred)
        p_t = y_pred * y + (1 - y_pred) * (1 - y)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean(1).sum()
