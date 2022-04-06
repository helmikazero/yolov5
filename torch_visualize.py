import torch
from torchsummary import summary

model = torch.load("runs/train/HelmetDetection/weights/best.pt")

summary(model)