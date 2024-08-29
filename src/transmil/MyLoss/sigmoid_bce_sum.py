import torch.nn as nn

class SigmoidBCESum(nn.Module):
    def __init__(self):
        super(SigmoidBCESum, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(
            reduction="none"
        )  # 'none' keeps the loss per element

    def forward(self, outputs, targets):
        outputs = self.sigmoid(outputs)
        loss = self.bce_loss(outputs, targets)
        loss = loss.sum()
        return loss
