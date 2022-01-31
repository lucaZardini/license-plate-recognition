import cv2
import numpy as np


class ImageProcessing:

    @staticmethod
    def to_gray_scale(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_filter(img: np.ndarray, filter: str) -> np.ndarray:
        if filter == "bilateral":
            return cv2.bilateralFilter(img, 11, 17, 17)

        if filter == "median":
            return cv2.medianBlur(img, 3)

        if filter == "gaussian":
            return cv2.GaussianBlur(img, (5, 5), 0)

        raise ValueError("Supported filters are: bilateral, median, gaussian")

    @staticmethod
    def apply_contrast_enhancement(img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    @staticmethod
    def apply_binarization(img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 3)

    @staticmethod
    def apply_canny_edge_detection(img: np.ndarray) -> np.ndarray:
        mean = img.mean()
        min_threshold = 0.66 * mean
        max_threshold = 1.33 * mean
        return cv2.Canny(img, min_threshold, max_threshold)

    @staticmethod
    def apply_dilation(img: np.ndarray, iterations: int = 1) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)

    @staticmethod
    def apply_erosion(img: np.ndarray, iterations: int = 1) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(img, kernel, iterations=iterations)