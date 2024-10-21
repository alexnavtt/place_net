import torch
from torch import Tensor

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # To prevent division by zero

    def forward(self, logits: Tensor, targets: Tensor):
        # Apply sigmoid to map the model outputs to the range [0, 1]
        outputs = torch.sigmoid(logits)
        
        # Flatten the tensors
        outputs = outputs.flatten()
        targets = targets.flatten()
        
        # Compute intersection and union
        intersection = (outputs * targets).sum()
        total = outputs.sum() + targets.sum()
        
        # Compute Dice loss
        dice_loss = 1 - (2 * intersection + self.smooth) / (total + self.smooth)
        return dice_loss
    
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, device='cuda:0'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0]).to(device))
        self.dice = DiceLoss()

    def forward(self, logits: Tensor, targets: Tensor):
        # Flatten the input
        targets = targets.flatten()

        # Get the BCE loss of the model output
        bce_loss = self.bce(logits.flatten(), targets.float())
        dice_loss = self.dice(logits.flatten(), targets.float())
    
        probabilities = torch.sigmoid(logits).flatten()
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        # Compute focal loss
        focal_loss = alpha_t * focal_factor * 0.5*(dice_loss + bce_loss)
        return focal_loss.sum()

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for false positives. Increasing this will decrease the
                   rate of false positives
            beta: Weight for false negatives. Increasing this will decrease the
                  rate of false negatives
            smooth: Smoothing factor to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor):
        """
        Args:
            logits: Predicted logits (before sigmoid), shape (N, *).
            targets: Ground truth binary labels, shape (N, *).

        Returns:
            torch.Tensor: Computed Tversky loss.
        """
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)

        # Flatten tensors to (N, -1)
        probs_flat = probabilities.flatten(start_dim=1)
        targets_flat = targets.flatten(start_dim=1)

        # Calculate True Positives, False Positives, and False Negatives
        TP = (probs_flat * targets_flat).sum(dim=1)
        FP = (probs_flat * (1 - targets_flat)).sum(dim=1)
        FN = ((1 - probs_flat) * targets_flat).sum(dim=1)

        # Compute Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Compute Tversky loss
        loss = 1 - tversky_index

        # Return the mean loss over the batch
        return loss.mean()