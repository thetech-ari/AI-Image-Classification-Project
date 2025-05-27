import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import urllib.request

# Load and preprocess image
def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Load labels for ImageNet classes
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_idx = []
    with urllib.request.urlopen(url) as f:
        class_idx = [line.decode('utf-8').strip() for line in f.readlines()]
    return class_idx

# Predict top-3 labels
def predict(image_tensor, model, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top3 = torch.topk(probs, 3)
        for idx, (i, p) in enumerate(zip(top3.indices, top3.values)):
            print(f"{idx+1}. {labels[i]} - {p.item()*100:.2f}%")

if __name__ == "__main__":
    img_path = "dog1.jpg"
    model = models.resnet50(pretrained=True)
    labels = load_labels()
    image_tensor = load_image(img_path)
    predict(image_tensor, model, labels)
