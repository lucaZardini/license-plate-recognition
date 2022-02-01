from __future__ import absolute_import, annotations

import os
from typing import Optional

import cv2
import imutils
import numpy as np

from utils.images import ImageUtils


class LicensePlateCandidate:
    def __init__(self, rect: any, img: np.ndarray, magnitude: Optional[float] = None) -> None:
        """
        License plate candidate model used to maintain the code readable
        :param rect: the rectangle which contains the contour
        :param img: the image of the candidate plate
        :param magnitude: the magnitude of the image
        """
        self.rect = rect
        self.img = img
        self.magnitude = magnitude


class LicensePlateDetection:

    @staticmethod
    def run_detection(source: str, destination: str, is_folder: bool) -> None:
        """
        This method runs the detection of the plate
        :param source: It can be a folder or a file. In case is a folder all the files in that folder are analyzed
        :param destination: the folder destination to save the analyzed image.
        :param is_folder: boolean which describe if the source is a folder or an image.
        """
        print("Running detection...")
        if not os.path.exists(destination):
            os.mkdir(destination)
        if is_folder:
            for file in os.listdir(source):
                LicensePlateDetection._extract_plate(source, file, destination)
        else:
            source, file = _extract_file(source)
            LicensePlateDetection._extract_plate(source, file, destination)
        print("Done")

    @staticmethod
    def _extract_plate(source: str, file: str, destination: str) -> None:
        """
        Private function that run the detection of the plate and store the result in the destination folder.
        :param source: The path of the folder containing the file
        :param file: The filename
        :param destination: The destination path
        """
        img = cv2.imread(source+"/"+file, cv2.IMREAD_COLOR)

        # Apply pre-processing
        gray = ImageUtils.to_gray_scale(img)
        filtered = ImageUtils.apply_filter(gray, "bilateral")
        contrast_enhanced = ImageUtils.apply_contrast_enhancement(filtered)

        work_img = contrast_enhanced.copy()

        # Get canny edges, apply dilation and closing
        canny_edges = ImageUtils.get_canny_edges(work_img)
        canny_edges = ImageUtils.apply_dilation(canny_edges, iterations=2)
        canny_edges = ImageUtils.apply_closing(canny_edges)

        # Find contours
        contours = cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Filter out contours based on criteria
        candidates = []
        for contour in contours:

            rect = cv2.boundingRect(contour)

            if _check_area(rect) and _check_ratio(rect):
                x, y, w, h = rect
                plate_img = work_img[y:y+h, x:x+w]

                if _check_color(plate_img):
                    candidate = LicensePlateCandidate(rect, plate_img)
                    candidates.append(candidate)

        # If no possible license plate, simply write the image without nothing
        if len(candidates) == 0:
            cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}_out.{file.rsplit('.')[1]}", img)
            return

        # Find the most probable license plate based on magnitude found with Sobel (only on vertical edges)
        max_magnitude = 0
        most_probable_license_plate = None
        for candidate in candidates:
            sobel_y = ImageUtils.get_sobel_y_edges(candidate.img)
            sobel_y_without_percentile = ImageUtils.remove_percentile(sobel_y, 85)

            magnitude = sobel_y_without_percentile.mean()
            candidate.magnitude = magnitude

            if magnitude > max_magnitude:
                max_magnitude = magnitude
                most_probable_license_plate = candidate

        # If present, draw rectangle and save image
        if most_probable_license_plate is not None:
            img = cv2.rectangle(
                img,
                most_probable_license_plate.rect,
                (255, 0, 0), 3
            )
            cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}_out.{file.rsplit('.')[1]}", img)
        else:
            cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}_out.{file.rsplit('.')[1]}", img)


def _extract_file(file_path: str) -> (str, str):
    """
    Try to get the file name from the filepath
    :param file_path: the path of the image
    :return: a tuple, the folder where the image is contained and the file name
    """
    slash_point = file_path.rfind("/")
    if slash_point == -1:
        return ".", file_path
    else:
        return file_path[:slash_point], file_path[slash_point+1:]


def _check_area(rect) -> bool:
    """
    Private function that verifies if the area of a rectangle is a valid one
    :param rect: The rectangle to calculate the area
    :return: The result of the verification
    """
    _, _, width, height = rect

    area = height * width
    if area < 1065 or area > 35000:
        return False

    return True


def _check_ratio(rect: any) -> bool:
    """
    Function that verifies if the rectangle has a valid ratio
    :param rect: The rectangle
    :return: The result of the verification
    """
    _, _, width, height = rect

    if height == 0:
        return False

    ratio = float(width) / float(height)
    if ratio < 1.5 or ratio > 5:
        return False

    return True


def _check_color(img: np.ndarray) -> bool:
    """
    FUnction that verifies if the image is white enough
    :param img: The image
    :return: The result of the verification
    """
    mean_value = img.mean()
    return mean_value >= 100
