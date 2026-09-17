import urllib.request

import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Load and preprocess the sample image
image = Image.open("week1/sample.jpg").convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Load ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(labels_url) as response:
    categories = [line.decode("utf-8").strip() for line in response.readlines()]

# Get top 3 predictions
top3_prob, top3_idx = torch.topk(probabilities, 3)

print("Top-3 predictions:")
for i in range(3):
    label = categories[top3_idx[i]]
    confidence = top3_prob[i].item() * 100
    print(f" {label:<25} {confidence:5.2f}%")
