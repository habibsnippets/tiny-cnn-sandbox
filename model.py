### Refactored Code
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
super(TinyCNN, self).__init__()

# Block 1: Conv -> BatchNorm -> ReLU -> MaxPool (28x28 -> 14x14)
self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
self.bn1 = nn.BatchNorm2d(16)

# Block 2: Conv -> BatchNorm -> ReLU -> MaxPool (14x14 -> 7x7)
self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
self.bn2 = nn.BatchNorm2d(32)

# Activation and Pooling
self.relu = nn.ReLU()
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Regularization & Classification Head
self.dropout = nn.Dropout(p=dropout_p)
self.fc1 = nn.Linear(32 * 7 * 7, 64)
self.fc2 = nn.Linear(64, num_classes)

def forward(self, x: torch.Tensor) -> torch.Tensor:
# Layer 1
x = self.conv1(x)
x = self.bn1(x)
x = self.relu(x)
x = self.pool(x)

# Layer 2
x = self.conv2(x)
x = self.bn2(x)
x = self.relu(x)
x = self.pool(x)

# Flatten (Batch Size, 32 * 7 * 7)
x = x.view(x.size(0), -1)

# Classifier
x = self.dropout(x)
x = self.fc1(x)
x = self.relu(x)
x = self.dropout(x)
x = self.fc2(x)

return x

