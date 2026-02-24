import torch
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.activations = None
        self.gradients = None

        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self):
        pooled = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i] *= pooled[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        return heatmap.detach().cpu().numpy()