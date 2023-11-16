import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms

# import model already trained for deeplabv3_resnet50
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

print(model)