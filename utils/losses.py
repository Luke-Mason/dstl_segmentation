import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lovasz_losses import lovasz_softmax
from sklearn.metrics import average_precision_score
import sys
epsilon = sys.float_info.epsilon


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2],
                                labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                      reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target, classes=output.size()[1])
        output = F.softmax(output, dim=1)
        # Check the device of the tensor
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        # Both tensors must be on the same device
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False,
                                           ignore_index=ignore_index,
                                           weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class JaccardCoefficient(nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super(JaccardCoefficient, self).__init__()

    def forward(self, output, target):
        # Calculate the intersection by element-wise multiplication and sum
        intersection = (target * output).sum()

        # Calculate the union by summing the individual sums of target and output
        union = target.sum() + output.sum() - intersection

        # Calculate the Jaccard coefficient (IoU)
        jaccard = (intersection + 1.0) / ((union + 1.0) + epsilon)

        return jaccard


class Recall(nn.Module):
    def __init__(self, ignore_index=255):
        super(Recall, self).__init__()

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = make_one_hot(target, classes=output.size()[1])

        true_positive = (output * target).sum(dim=(2, 3))
        actual_positive = target.sum(dim=(2, 3))

        recall = true_positive / (
                actual_positive + 1e-10)  # Small epsilon to avoid division by zero

        return recall.mean()


class Precision(nn.Module):
    def __init__(self, ignore_index=255):
        super(Precision, self).__init__()

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = make_one_hot(target, classes=output.size()[1])

        true_positive = (output * target).sum(dim=(2, 3))
        predicted_positive = output.sum(dim=(2, 3))

        precision = true_positive / (
                predicted_positive + 1e-10)  # Small epsilon to avoid division by zero

        return precision.mean()


class F1Score(nn.Module):
    def __init__(self, ignore_index=255):
        super(F1Score, self).__init__()
        self.precision = Precision()
        self.recall = Recall()

    def forward(self, output, target):
        precision = self.precision(output, target)
        recall = self.recall(output, target)
        f1 = 2 * (precision * recall) / (
                precision + recall + 1e-10)  # Small epsilon to avoid division by zero

        return f1.mean()


class MeanAveragePrecision(nn.Module):
    def __init__(self, ignore_index=255):
        super(MeanAveragePrecision, self).__init__()

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = make_one_hot(target, classes=output.size()[1])

        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        ap = average_precision_score(target_flat.cpu().numpy(),
                                     output_flat.cpu().numpy())

        return torch.tensor(ap).to(output.device)


class BCELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)


class MSELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        return self.mse_loss(output, target)


class DiceLoss2(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        eps = 1e-10
        output = torch.sigmoid(output)
        intersection = (output * target).sum(dim=(2, 3))
        union = (output + target).sum(dim=(2, 3)) + eps
        dice = 2 * intersection / union
        return 1 - dice.mean()


class BCE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        dice_loss = self.dice(output, target)
        return bce_loss + dice_loss


class BCE_JaccardLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_JaccardLoss, self).__init__()
        self.smooth = smooth
        self.jaccard = JaccardCoefficient()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        jaccard_loss = self.jaccard(output, target)
        return bce_loss + jaccard_loss


class BCE_F1Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_F1Loss, self).__init__()
        self.smooth = smooth
        self.f1 = F1Score()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        f1_loss = self.f1(output, target)
        return bce_loss + f1_loss


# class BCE_MeanAveragePrecisionLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255,
#                  weight=None):
#         super(BCE_MeanAveragePrecisionLoss, self).__init__()
#         self.smooth = smooth
#         self.map = MeanAveragePrecision()
#         self.bce = BCELoss()
#
#     def forward(self, output, target):
#         bce_loss = self.bce(output, target)
#         map_loss = self.map(output, target)
#         return bce_loss + map_loss


class Dice_JaccardLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(Dice_JaccardLoss, self).__init__()
        self.smooth = smooth
        self.jaccard = JaccardCoefficient()
        self.dice = DiceLoss()

    def forward(self, output, target):
        dice_loss = self.dice(output, target)
        jaccard_loss = self.jaccard(output, target)
        return dice_loss + jaccard_loss


class Dice_F1Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(Dice_F1Loss, self).__init__()
        self.smooth = smooth
        self.f1 = F1Score()
        self.dice = DiceLoss()

    def forward(self, output, target):
        dice_loss = self.dice(output, target)
        f1_loss = self.f1(output, target)
        return dice_loss + f1_loss


# class Dice_MeanAveragePrecisionLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255,
#                  weight=None):
#         super(Dice_MeanAveragePrecisionLoss, self).__init__()
#         self.smooth = smooth
#         self.map = MeanAveragePrecision()
#         self.dice = DiceLoss()
#
#     def forward(self, output, target):
#         dice_loss = self.dice(output, target)
#         map_loss = self.map(output, target)
#         return dice_loss + map_loss


class Jaccard_F1Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(Jaccard_F1Loss, self).__init__()
        self.smooth = smooth
        self.f1 = F1Score()
        self.jaccard = JaccardCoefficient()

    def forward(self, output, target):
        jaccard_loss = self.jaccard(output, target)
        f1_loss = self.f1(output, target)
        return jaccard_loss + f1_loss


# class Jaccard_MeanAveragePrecisionLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255,
#                  weight=None):
#         super(Jaccard_MeanAveragePrecisionLoss, self).__init__()
#         self.smooth = smooth
#         self.map = MeanAveragePrecision()
#         self.jaccard = JaccardCoefficient()
#
#     def forward(self, output, target):
#         jaccard_loss = self.jaccard(output, target)
#         map_loss = self.map(output, target)
#         return jaccard_loss + map_loss


# class F1_MeanAveragePrecisionLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255,
#                  weight=None):
#         super(F1_MeanAveragePrecisionLoss, self).__init__()
#         self.smooth = smooth
#         self.map = MeanAveragePrecision()
#         self.f1 = F1Score()
#
#     def forward(self, output, target):
#         f1_loss = self.f1(output, target)
#         map_loss = self.map(output, target)
#         return f1_loss + map_loss


class BCE_Dice_JaccardLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_Dice_JaccardLoss, self).__init__()
        self.smooth = smooth
        self.jaccard = JaccardCoefficient()
        self.dice = DiceLoss()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        dice_loss = self.dice(output, target)
        jaccard_loss = self.jaccard(output, target)
        return bce_loss + dice_loss + jaccard_loss


class BCE_Dice_F1Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_Dice_F1Loss, self).__init__()
        self.smooth = smooth
        self.f1 = F1Score()
        self.dice = DiceLoss()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        dice_loss = self.dice(output, target)
        f1_loss = self.f1(output, target)
        return bce_loss + dice_loss + f1_loss


# class BCE_Dice_MeanAveragePrecisionLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255,
#                  weight=None):
#         super(BCE_Dice_MeanAveragePrecisionLoss, self).__init__()
#         self.smooth = smooth
#         self.map = MeanAveragePrecision()
#         self.dice = DiceLoss()
#         self.bce = BCELoss()
#
#     def forward(self, output, target):
#         bce_loss = self.bce(output, target)
#         dice_loss = self.dice(output, target)
#         map_loss = self.map(output, target)
#         return bce_loss + dice_loss + map_loss


class BCE_Jaccard_F1Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(BCE_Jaccard_F1Loss, self).__init__()
        self.smooth = smooth
        self.f1 = F1Score()
        self.jaccard = JaccardCoefficient()
        self.bce = BCELoss()

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        jaccard_loss = self.jaccard(output, target)
        f1_loss = self.f1(output, target)
        return bce_loss + jaccard_loss + f1_loss
