from __future__ import absolute_import, annotations

import os
from dataclasses import dataclass

import cv2
import imutils
import numpy as np


@dataclass
class Contour:
        size: int
        shape: (int, int)
        contour: np.ndarray
        magnitude: int


class CannyEdge:

    @staticmethod
    def run_detection(source: str, destination: str, is_folder: bool) -> None:
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

        # Apply resize
        img = cv2.resize(img, (620, 480))

        # Transform the image in gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        # Take the edge
        edged = cv2.Canny(gray, 30, 200)

        # Apply the dilatation operation
        kernel = np.ones((3,3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        # Find contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Sort contours from higher to lower
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        contours_list = []

        for c in contours:

            # Define a polygon for the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            # If the polygon has 4 sides
            if len(approx) == 4:
                contours_list.append(approx)

        if len(contours_list) == 0:
            detected = 0
            print(f"No contour detected for file {file}")
        else:
            detected = 1

        if detected == 1:
            valid_contours_list = []
            max_dimension = 0
            max_magnitude = 0

            # Take only the 10 biggest contours
            for contours_abc in contours_list[:10]:

                # Cut the plate section
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contours_abc], 0, 255, -1, )
                cv2.bitwise_and(img, img, mask=mask)
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                candidate = gray[topx:bottomx + 1, topy:bottomy + 1]

                # Calculate the gradient of the candidate image
                gX = cv2.Sobel(candidate, cv2.CV_64F, 1, 0)
                gY = cv2.Sobel(candidate, cv2.CV_64F, 0, 1)
                magnitude=np.sqrt((gX ** 2) + (gY ** 2)).mean()

                # Create new class for contour
                contour = Contour(size=candidate.size, shape=candidate.shape, contour=contours_abc, magnitude=magnitude)
                max_dimension = max(max_dimension, contour.size)
                max_magnitude = max(max_magnitude, np.sqrt((gX ** 2) + (gY ** 2)).mean())

                valid_contours_list.append(contour)

            weight = 0.4
            # Sort the contours based on the size and the gradient weighted
            valid_contours_list = sorted(valid_contours_list, key=lambda x: (
                        weight * (x.size / max_dimension) + (1 - weight) * (x.magnitude / max_magnitude)),
                               reverse=True)
            i = 0

            # TODO: when to stop the print
            for contour in valid_contours_list:
                h, w = contour.shape
                # If the shape of the contour is not similar as a plate, discharge the contour.
                if w / h < 0.66 or w / h > 4:
                    continue

                cv2.drawContours(img, [contour.contour], -1, (0, 0, 255), 3)
                cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}{str(i)}.{file.rsplit('.')[1]}", img)
                i = i + 1


def _extract_file(file_path: str) -> (str, str):
    slash_point = file_path.rfind("/")
    return file_path[:slash_point], file_path[slash_point+1:]
