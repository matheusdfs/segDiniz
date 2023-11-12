import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms

# import model already trained for deeplabv3_resnet50
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

# load image used in the inference
img = Image.open("/Users/matheusdiniz/Documents/segDiniz/data/cityscape/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000543_000019_leftImg8bit.png")

# define transformations to image and apply it
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = transform(img)

# add img_batch field in the original image
img_batch = torch.unsqueeze(img, 0)

# perform the inference
model.eval()
with torch.no_grad():
    outputs = model(img_batch)

prediction = outputs['out'].squeeze(0).cpu().numpy()
prediction = np.argmax(prediction, axis=0)

plt.imshow(prediction, interpolation='nearest')
plt.show()