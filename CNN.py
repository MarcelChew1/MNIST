import idx2numpy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

train_image_file = 'images/train-images.idx3-ubyte'
train_label_file = 'images/train-labels.idx1-ubyte'
test_image_file = 'images/t10k-images.idx3-ubyte'
test_label_file = 'images/t10k-labels.idx1-ubyte'

train_images = torch.tensor(idx2numpy.convert_from_file(train_image_file) // 250, dtype=torch.float32)
train_labels = torch.tensor(idx2numpy.convert_from_file(train_label_file), dtype=torch.long)
test_images = torch.tensor(idx2numpy.convert_from_file(test_image_file) // 250, dtype=torch.float32)
test_labels = torch.tensor(idx2numpy.convert_from_file(test_label_file), dtype=torch.long)

batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 28
out_size = 10
max_iters = 15000
eval_interval = 500
eval_iters = 200
n_layer = 3
kernel_size = 100
dropout = 0.2
learning_rate = 1e-4

torch.manual_seed(983031)
# plt.imshow(image1, cmap=plt.cm.binary)

# convert images into 0, 1 encoding as inputs
# labels are numbered 0 - 9 
train_images = train_images.view(-1, 1, image_size, image_size)
test_images = test_images.view(-1, 1, image_size, image_size)
print(train_images.shape)
# first develop a convolutional neural network

def get_batch(split):
  if split == 'train':
    images = train_images
    labels = train_labels
  else:
    images = test_images
    labels = test_labels

  idx = torch.randint(len(images), (batch_size,))  
  x = torch.stack([images[i] for i in idx]) 
  y = torch.stack([labels[i] for i in idx]) 
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      X, Y = get_batch(split)

      logits, loss = model(X, Y)
      losses[k] = loss.item()
    
    out[split] = losses.mean()
  model.train()
  return out

class Block(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(n_out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

  def forward(self, x):
    return self.net(x)

# basically need to convert 28 * 28 into a value from 0 - 9 
# so out values are going to be B * 
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolutions = nn.Sequential(*[Block(max(1, i*kernel_size), (i+1)*kernel_size) for i in range(n_layer)])
    self.final = nn.Linear(n_layer*kernel_size * (image_size // (2**n_layer)) * (image_size // (2**n_layer)), out_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, targets=None):

    x = self.convolutions(x)
    # print(x.shape)

    x = self.dropout(x)
    x = x.view(batch_size, -1) # flatten

    logits = self.final(x)
    if targets == None:
      loss = None
    else: 
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  


model = CNN()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


