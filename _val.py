from _init import *

def pixel_accuracy(output, mask):
    with torch.no_grad():
        #  1. Convert the output to class predictions using argmax after applying softmax
        preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
        #  2. Create a tensor of binary values indicating correct predictions
        correct = (preds == mask).float()
        #  3. Calculate accuracy by dividing the number of correct predictions by the total number of predictions
        accuracy = correct.sum() / correct.numel()
    return accuracy

def mIoU(pred_mask, mask, n_classes=3, ignore_class=0):
    """
    For batch input, calculates per-image mIoU and averages across the batch.
    """
    with torch.no_grad():
        probs = F.softmax(pred_mask, dim=1)
        preds = torch.argmax(probs, dim=1)  # (B, H, W)

        batch_size = preds.shape[0]
        batch_miou_scores = []

        # Calculate mIoU for each image in the batch (per-image averaging)
        for b in range(batch_size):
            pred_img = preds[b]  # (H, W)
            mask_img = mask[b]   # (H, W)
            iou_list = []

            for cls in range(n_classes):
                if cls == ignore_class: continue
                pred_c = (pred_img == cls)
                label_c = (mask_img == cls)
                intersection = (pred_c & label_c).sum().float()
                union = (pred_c | label_c).sum().float()

                # Skip if union is 0 (class doesn't exist in both GT and Pred)
                if union == 0: continue
                iou = intersection / union
                iou_list.append(iou)

            # Calculate mIoU for this image
            if len(iou_list) > 0:
                img_miou = torch.stack(iou_list).mean()
                batch_miou_scores.append(img_miou)

        # Return mean of per-image mIoU scores
        return float(torch.stack(batch_miou_scores).mean().item()) if len(batch_miou_scores) else 0.0

class DiceLoss(nn.Module):
    """
    Dice loss
    """

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Calculation of dice loss

        :param inputs: model predictions
        :param targets: target values
        :param eps: stability factor, defaults to 1e-6
        :return: loss value
        """
        inputs = torch.softmax(inputs, dim=1).clamp(self.eps, 1.0 - self.eps)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=3)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = torch.sum(inputs * targets_onehot, dims)
        union = torch.sum(inputs + targets_onehot, dims)

        denominator = torch.clamp(union + self.eps, min=1e-6)
        dice = (2 * intersection) / denominator
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    """
    def __init__(self, alpha=None, gamma=2.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) raw logits
        targets: (N, H, W) integer class labels (0 ~ C-1)
        """
        # Cross entropy per pixel
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape: (N, H, W)
        # pt = torch.exp(-ce_loss)
        # focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        pt = torch.exp(-ce_loss).clamp(self.eps, 1.0 - self.eps)
        focal_loss = (1 - pt) * (1 - pt) * ce_loss

        # Apply class weighting (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()