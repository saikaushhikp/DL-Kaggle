import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class MultiSpectralDataset(Dataset):
    def __init__(self, X, Y, indices, augment=False, mean=None, std=None, compute_stats=False):
        """
        X, Y          : mmap arrays or numpy arrays
        indices       : list of sample indices
        mean, std     : optional precomputed normalization stats
        compute_stats : if True -> compute mean/std from X[indices]
        """
        self.X = X
        self.Y = Y
        self.indices = indices
        self.augment = augment

        # -------------------------------------------------------
        # Compute normalization stats (ONLY ON TRAINING SET)
        # -------------------------------------------------------
        if compute_stats:
            C = X.shape[1]
            mean = np.zeros(C, dtype=np.float64)
            M2   = np.zeros(C, dtype=np.float64)
            count = 0

            for idx in indices:
                x = X[idx].astype(np.float32)     # convert to float32
                pixels = x.reshape(C, -1)
                count_new = pixels.shape[1]

                # incremental mean/std update (Welford algorithm)
                delta = pixels.mean(axis=1) - mean
                mean += delta * (count_new / (count + count_new))

                M2 += ((pixels - mean[:, None])**2).sum(axis=1)

                count += count_new

            self.mean = mean.astype(np.float32)
            self.std = np.sqrt(M2 / count).astype(np.float32) + 1e-6

        else:
            self.mean = mean       # use externally provided stats
            self.std  = std
        # -------------------------------------------------------

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        #  Load image 
        img = torch.tensor(self.X[idx], dtype=torch.float32)

        # Apply normalization if available 
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=torch.float32)[:, None, None]
            std  = torch.tensor(self.std, dtype=torch.float32)[:, None, None]
            img = (img - mean) / std

        # TEST MODE 
        if self.Y is None:
            return img

        # TRAIN / VAL MODE 
        mask = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(0)

        return img, mask
    
    
##############################
# IoU Metric and Dice Loss
##############################
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred (torch.Tensor): Predicted logits of shape (B, 1, H, W)
        target (torch.Tensor): Ground truth mask of shape (B, 1, H, W)
        smooth (float): Smoothing constant to avoid division by zero
    Returns:
        torch.Tensor: Scalar Dice Loss
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities

    # Flatten each image in batch to compute per-sample Dice
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)

    # Return mean Dice loss over batch
    return 1 - dice.mean()


def combined_loss(outputs, target, alpha=0.7):
    """
    outputs can be either:
     - single tensor (deep_supervision=False)
     - tuple: (main, ds3, ds4)
    """

    if isinstance(outputs, tuple):
        main, ds3, ds4 = outputs

        loss_main = alpha * bce_loss(main, target) + (1 - alpha) * dice_loss(main, target)
        loss_ds3  = alpha * bce_loss(ds3, target)  + (1 - alpha) * dice_loss(ds3, target)
        loss_ds4  = alpha * bce_loss(ds4, target)  + (1 - alpha) * dice_loss(ds4, target)

        # Weighted deep supervision
        return loss_main + 0.5 * loss_ds3 + 0.25 * loss_ds4

    else:
        # Single output
        return alpha * bce_loss(outputs, target) + (1 - alpha) * dice_loss(outputs, target)


def iou_score(pred, gt, n_classes=2, eps=1e-10):

    #  Handle MFNet deep supervision: get main output
    if isinstance(pred, tuple):
        pred = pred[0]

    with torch.no_grad():
        # Binary segmentation: pred shape = (B,1,H,W)
        if pred.shape[1] == 1:
            pred = (pred > 0.1).long()  # threshold at 0.1
        else:
            pred = torch.argmax(pred, dim=1, keepdim=True)

        pred = pred.squeeze(1)
        gt = gt.squeeze(1)

        iou_per_class = []

        for cls in range(n_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)

            intersection = (pred_cls & gt_cls).sum().float()
            union = (pred_cls | gt_cls).sum().float()

            if union == 0:
                iou_per_class.append(float('nan'))
            else:
                iou = (intersection + eps) / (union + eps)
                iou_per_class.append(iou.item())

        return np.nanmean(iou_per_class)    