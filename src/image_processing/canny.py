from __future__ import absolute_import, annotations

import os
from dataclasses import dataclass

import cv2
import imutils
import numpy as np

from src.utils.preprocessing import ImageProcessing


@dataclass
class Contour:
    size: int
    shape: (int, int)
    contour: np.ndarray
    magnitude: int


class CannyEdge:

    @staticmethod
    def run_detection(source: str, destination: str, is_folder: bool) -> None:
        if not os.path.exists(destination):
            os.mkdir(destination)
        if is_folder:
            for file in os.listdir(source):
                CannyEdge._extract_plate(source, file, destination)
        else:
            source, file = _extract_file(source)
            CannyEdge._extract_plate(source, file, destination)

    @staticmethod
    def _extract_plate(source: str, file: str, destination: str) -> None:
        # Read the image
        img = cv2.imread(source+"/"+file, cv2.IMREAD_COLOR)

        # Apply pre-processing
        gray = ImageProcessing.to_gray_scale(img)
        gray = ImageProcessing.apply_filter(gray, "bilateral")
        gray = ImageProcessing.apply_filter(gray, "gaussian")
        gray = ImageProcessing.apply_contrast_enhancement(gray)
        edged = ImageProcessing.apply_canny_edge_detection(gray)
        edged = ImageProcessing.apply_dilation(edged, iterations=1)

        # Find contours
        contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Sort contours from higher to lower based on area
        all_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Extract all four sided contours
        four_sided_contours = []
        for c in all_contours:
            # Define a polygon for the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If the polygon has 4 sides, consider it a possible plate
            if len(approx) == 4:
                four_sided_contours.append(approx)

        valid_contours_list = []
        max_dimension = 0
        max_magnitude = 0
        # Compute gradient along Y direction for each of the first 10 four sided contours
        for contour in four_sided_contours[:10]:
            # Cut the plate section
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            cv2.bitwise_and(img, img, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            candidate = gray[topx:bottomx + 1, topy:bottomy + 1]

            # Calculate the gradient along Y of the candidate area
            gY = cv2.Sobel(candidate, cv2.CV_64F, 0, 1)
            magnitude = gY.mean()

            # Create new class for contour
            my_contour = Contour(size=cv2.contourArea(contour), shape=candidate.shape, contour=contour, magnitude=magnitude)
            max_dimension = max(max_dimension, my_contour.size)
            max_magnitude = max(max_magnitude, magnitude)

            valid_contours_list.append(my_contour)

        # Trade off between dimension of contour and magnitude
        weight = 0.4
        valid_contours_list = sorted(valid_contours_list, reverse=True, key=lambda x: (
                weight * (x.size / max_dimension) + (1 - weight) * (x.magnitude / max_magnitude)
        ))

        # Take the first contour that fits the aspect ratio of a license plate
        for count, contour in enumerate(valid_contours_list):
            h, w = contour.shape
            ratio = w / h
            # If the shape of the contour is not similar to that of a plate, discard the contour.
            if 3 < ratio < 6:
                cv2.drawContours(img, [contour.contour], -1, (0, 0, 255), 3)
                break

        # Save the image with all contours
        cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}_out.{file.rsplit('.')[1]}", img)


def _extract_file(file_path: str) -> (str, str):
    slash_point = file_path.rfind("/")
    if slash_point == -1:
        return ".", file_path
    else:
        return file_path[:slash_point], file_path[slash_point+1:]
