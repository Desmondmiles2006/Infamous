import cv2

class CameraStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()