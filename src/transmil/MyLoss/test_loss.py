import torch
import torch.nn as nn


class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(
            reduction="none"
        )  # 'none' keeps the loss per element

    def forward(self, outputs, targets):
        # Apply sigmoid activation
        outputs = self.sigmoid(outputs)

        # Calculate binary cross-entropy loss for each label
        loss = self.bce_loss(outputs, targets)

        # Sum the losses for all labels
        loss = loss.sum()

        return loss


# Example usage
if __name__ == "__main__":
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output = loss(input, target)
    output.backward()

    # Simulated outputs (logits) and targets (binary labels)
    outputs = torch.tensor([[0.5, 1.5, -1.0], [0.3, -0.8, 2.0]], requires_grad=True)
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)

    # Initialize the custom loss function
    custom_loss_fn = CustomBCELoss()

    # Compute the loss
    print(outputs, targets)
    loss = custom_loss_fn(outputs, targets)

    print(f"Loss: {loss.item()}")

    # Backward pass (optional, if you want to update model weights)
    loss.backward()
