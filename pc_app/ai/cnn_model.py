import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class CNNClassifier:
    def __init__(self, device):
        weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.eval().to(device)

        self.device = device
        self.classes = ["Normal", "Abnormal"]

    def predict(self, input_tensor):
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        cls = torch.argmax(probs, dim=1).item()
        conf = probs[0][cls].item()
        return cls, conf, output