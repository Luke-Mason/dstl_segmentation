import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from sklearn.metrics import average_precision_score
epsilon = sys.float_info.epsilon
def pixel_accuracy(correct_pixels, total_labeled_pixels):
    return correct_pixels / (total_labeled_pixels + epsilon)

def precision(intersection, predicted_positives):
    return intersection / (predicted_positives + epsilon)

def recall(intersection, total_positives):
    return intersection / (total_positives + epsilon)

def f1_score(intersection, predicted_positives, total_positives):
    p = precision(intersection, predicted_positives)
    r = recall(intersection, total_positives)

    # Compute F1 score
    return 2 * (p * r) / (p + r + epsilon)

def mean_average_precision(average_precision):
    return np.mean(average_precision)

def intersection_over_union(intersection, union):
    return intersection / (union + epsilon)

def eval_metrics(o, t, threshold=0.5):
    output = (o > threshold).float().to(torch.int)
    target = (t > threshold).float().to(torch.int)

    # All positives in prediction
    predicted_positives = torch.sum(output)

    # Pixel Accuracy Components
    # Correct pixels
    correct_pixels = torch.sum(output == target)
    total_pixels = output.numel()

    # Recall Components
    total_positives = torch.sum(target)

    # IoU Components
    # True positives
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target) - intersection

    # # Average Precision Components
    # # Number of classes
    # num_classes = output.shape[1]
    # # Initialize average precision list
    # average_precision = []
    #
    # # For each class
    # for class_ in range(num_classes):
    #     # Get class-specific output and target
    #     output_class = output[:, class_, :, :]
    #     target_class = target[:, class_, :, :]
    #
    #     # Get class-specific average precision
    #     class_average_precision = average_precision_score(
    #         target_class.flatten().cpu(),
    #         output_class.flatten().cpu())
    #
    #     # Append to the list of average precisions
    #     average_precision.append(class_average_precision)

    return {
        # Also true_positives
        "intersection": intersection.item(),
        "union": union.item(),
        "total_positives": total_positives.item(),
        "total_pixels": total_pixels,
        "correct_pixels": correct_pixels.item(),
        "predicted_positives": predicted_positives.item(),
        # "average_precision": average_precision,
    }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)
