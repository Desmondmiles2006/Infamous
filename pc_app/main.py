import cv2
import torch
import numpy as np

from camera.camera_stream import CameraStream
from ai.cnn_model import CNNClassifier
from xai.gradcam import GradCAM
from cv.cell_detection import detect_cell_borders
from utils.preprocessing import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

camera = CameraStream("http://192.168.1.3:8080/video")
classifier = CNNClassifier(device)
gradcam = GradCAM(classifier.model)

while True:
    frame = camera.read()
    if frame is None:
        continue

    h, w, _ = frame.shape

    input_tensor = transform(frame).unsqueeze(0).to(device)
    cls, conf, output = classifier.predict(input_tensor)

    classifier.model.zero_grad()
    output[0, cls].backward()

    heatmap = gradcam.generate()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    cv2.putText(
        overlay,
        f"{classifier.classes[cls]} ({conf:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    border_frame, count = detect_cell_borders(frame)
    cv2.putText(
        border_frame,
        f"Cells Detected: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Raw Microscope Feed", frame)
    cv2.imshow("Grad-CAM Heatmap", heatmap)
    cv2.imshow("AI Overlay (Explainable Output)", overlay)
    cv2.imshow("Cell Boundary Detection", border_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()