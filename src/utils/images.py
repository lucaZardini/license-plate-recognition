import cv2
import numpy as np


class ImageUtils:

    @staticmethod
    def to_gray_scale(img: np.ndarray) -> np.ndarray:
        """
        Apply the gray scale to the image
        :param img: The initial image
        :return: the gray scale image
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_filter(img: np.ndarray, filter: str) -> np.ndarray:
        """
        Apply filter to the image
        :param img: The image to process
        :param filter: Possible filters are: bilateral, median, gaussian
        :return: The image filtered
        """
        if filter == "bilateral":
            return cv2.bilateralFilter(img, 15, 69, 69)

        if filter == "median":
            return cv2.medianBlur(img, 3)

        if filter == "gaussian":
            return cv2.GaussianBlur(img, (3, 3), 0)

        raise ValueError("Supported filters are: bilateral, median, gaussian")

    @staticmethod
    def apply_contrast_enhancement(img: np.ndarray) -> np.ndarray:
        """
        Apply contrast enhancement to the image
        :param img: The image to process
        :return: The image after the contrast enhancement processing
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    @staticmethod
    def get_canny_edges(img: np.ndarray) -> np.ndarray:
        """
        Apply canny operator to the image
        :param img: The image
        :return: Image after the application of canny operator
        """
        mean = img.mean()
        min_threshold = 0.66 * mean
        max_threshold = 3 * min_threshold
        return cv2.Canny(img, min_threshold, max_threshold)

    @staticmethod
    def get_sobel_y_edges(img: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator
        :param img: The image
        :return: The image after the application of Sobel operator
        """
        gradient_y = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_gradient_y = cv2.convertScaleAbs(gradient_y)
        return abs_gradient_y

    @staticmethod
    def remove_percentile(img: np.ndarray, percentile: int) -> np.ndarray:
        """
        Remove percentile to the image
        :param img: The image to apply the function
        :param percentile: Scalar
        :return: The image after the operation
        """
        pixel_values = []
        for img_row in img:
            for img_col in img_row:
                pixel_values.append(img_col)

        percentile = np.percentile(pixel_values, percentile)

        img_without_percentile = img.copy()
        for i, img_row in enumerate(img):
            for j, img_col in enumerate(img_row):
                if img_col <= percentile:
                    img_without_percentile[i, j] = 0

        return img_without_percentile

    @staticmethod
    def apply_closing(img: np.ndarray) -> np.ndarray:
        """
        Apply the closing morphological operation
        :param img: The starting image
        :return: The modified image
        """
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def apply_opening(img: np.ndarray) -> np.ndarray:
        """
        Apply the opening morphological operation
        :param img: The starting image
        :return: The modified image
        """
        kernel = np.ones(3, np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def apply_dilation(img: np.ndarray, iterations: int = 1, size: int = 3) -> np.ndarray:
        """
        Apply the dilation morphological operation
        :param img: The starting image
        :param iterations: The number of times to iterate the dilation
        :param size: The size of the kernel
        :return: The modified image
        """
        return cv2.dilate(img, (size, size), iterations=iterations)

    @staticmethod
    def apply_erosion(img: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply the erosion morphological operation
        :param img: The starting image
        :param iterations: THe number of times to iterate the erosion
        :return: The modified image
        """
        return cv2.erode(img, (3, 3), iterations=iterations)
