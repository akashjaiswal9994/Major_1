import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
#from PLI import Image


model = models.vgg16(pretrained=True).features

print(model)
