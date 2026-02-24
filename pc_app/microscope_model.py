# microscope_model.py

import cv2
import numpy as np

class MicroscopeAnalyzer:
    """
    ML-based microscope slide analyzer
    """

    def __init__(self):
        self.min_area = 100
        self.max_area = 5000

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, blurred

    def focus_score(self, gray):
        """
        Variance of Laplacian as focus metric
        """
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect_cells(self, blurred):
        """
        Detect cell-like structures using adaptive thresholding
        """
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                cells.append(cnt)

        return cells, thresh

    def analyze(self, frame):
        """
        Main ML inference pipeline
        """
        gray, blurred = self.preprocess(frame)
        focus = self.focus_score(gray)
        cells, mask = self.detect_cells(blurred)

        result = {
            "focus_score": round(focus, 2),
            "cell_count": len(cells),
            "cells": cells,
            "mask": mask
        }

        return result
