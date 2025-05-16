import cv2
import numpy as np


def get_features(path):
    """thumbnail -> numbers the model can learn from"""
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = gray.mean()
    contrast = gray.std()
    saturation = hsv[:, :, 1].mean()

    return [brightness, contrast, saturation]
