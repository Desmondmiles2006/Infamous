import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load CNN Model
# -------------------------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.eval().to(device)

classes = ["Normal", "Abnormal"]

# -------------------------------
# Grad-CAM Hooks
# -------------------------------
activations, gradients = None, None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------------------------------
# Preprocessing
# -------------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Grad-CAM
# -------------------------------
def generate_gradcam():
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-8

    return heatmap.detach().cpu().numpy()

# -------------------------------
# Cell Border Detection (Classical CV)
# -------------------------------
def detect_cell_borders(frame):
    """
    Explainable, deterministic cell boundary extraction
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold works well for microscopy
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    border_frame = frame.copy()
    cell_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # noise filtering
            cv2.drawContours(border_frame, [cnt], -1, (0, 255, 255), 1)
            cell_count += 1

    return border_frame, cell_count

# -------------------------------
# Webcam / Microscope Feed
# -----------------------import cv2

#cap = cv2.VideoCapture(1)  # or 2, based on step 2

address = "http://192.168.1.3:8080/video"
cap = cv2.VideoCapture(address)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ---------- CNN INFERENCE ----------
    resized = cv2.resize(frame, (224, 224))
    input_tensor = transform(resized).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    model.zero_grad()
    output[0, pred_class].backward()

    # ---------- GRAD-CAM ----------
    heatmap = generate_gradcam()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ---------- OVERLAY ----------
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    label = f"{classes[pred_class]} ({confidence:.2f})"
    cv2.putText(overlay, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ---------- CELL BORDER FEED ----------
    border_frame, cell_count = detect_cell_borders(frame)
    cv2.putText(border_frame, f"Cells Detected: {cell_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ---------- DISPLAY WINDOWS ----------
    cv2.imshow("Raw Microscope Feed", frame)
    cv2.imshow("Grad-CAM Heatmap", heatmap_color)
    cv2.imshow("AI Overlay (Explainable Output)", overlay)
    cv2.imshow("Cell Boundary Detection", border_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
