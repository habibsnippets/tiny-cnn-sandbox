import torch
imoprt torch.nn as nn
import torch.optim as optim

# deifne a tiny cnn for cpu training

class TinyCNN(nn.Module):
  def __init__(self):
    super(TinyCNN, self).__init__()
    self.conv1=nn.Conv2d(1,8,kernel_size=3,stride=1, padding=1)
    self.relu=nn.ReLU()
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1=nn.Linear(8*12*14,10)

  def forward(self,x):
    x=self.conv1(x)
    x=self.relu(x)
    x=self.pool(x)
    x=x.view(x.size(0),-1) #flattening
    return x 

if __name__ == "__main__":
  model = TinyCNN()
  print("model initialized successfully")
  dummy_input = torch.randn(1,1,28,28)
  output = model(dummy_input)
  print(f"output shape is : {output.shape")


