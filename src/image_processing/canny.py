from __future__ import absolute_import, annotations

import os

import cv2
import imutils
import numpy as np


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
        img = cv2.imread(source+"/"+file, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (620, 480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        edged = cv2.Canny(gray, 30, 200)

        kernel = np.ones((3,3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        # cv2.imwrite("/home/luca/Scaricati/targa-auto-panda0.jpg", edged)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        contours_list = []

        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                contours_list.append(approx)

        if len(contours_list) == 0:
            detected = 0
            print(f"No contour detected for file {file}")
        else:
            detected = 1

        if detected == 1:
            i = 100
            for contours_abc in contours_list[:10]:
                cv2.drawContours(img, [contours_abc], -1, (0, 0, 255), 3)
                cv2.imwrite(f"{destination}/{file}{str(i)}", img)
                i=i+1


def _extract_file(file_path: str) -> (str, str):
    slash_point = file_path.rfind("/")
    return file_path[:slash_point], file_path[slash_point+1:]
