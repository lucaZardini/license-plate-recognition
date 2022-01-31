import cv2
import imutils
import numpy as np

from text_detector.text_detector import funziona

source = "/home/luca/Scaricati"
file = "targhe-verdi-auto-elettrica-1.jpg"
destination = "/home/luca/Scaricati"
print(source+"/"+file)
img = cv2.imread(source + "/" + file, cv2.IMREAD_COLOR)

img = cv2.resize(img, (620, 480))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200)

kernel = np.ones((3, 3), np.uint8)
edged = cv2.dilate(edged, kernel, iterations=1)
cv2.imwrite(destination+"/"+"cheneso.jpeg", edged)

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
    test = []
    max_dimension = 0
    max_magnitude = 0
    for contours_abc in contours_list[:10]:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contours_abc], 0, 255, -1, )
        cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        # cv2.imwrite(f"{destination}/{file.rsplit('.')[0]}cutted{file.rsplit('.')[1]}", Cropped)
        if funziona(Cropped):
            cv2.drawContours(img, [contours_abc], -1, (0, 0, 255), 3)