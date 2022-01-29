import cv2
import matplotlib.pyplot as plt
import sys

# To run it, pass as input the file path.

image_path = sys.argv[1]
haar_path = '/home/luca/Documenti/Università/AIS/1Semestre/signalImageVideo/car-plate-recognition/src/image_processing/Cascade_Haar/haar.xml'


#function that shows the image
def display(img, out_file):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_file, img)


def detect_plate(img):
    plate_img = img.copy()

    # gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=10)

    # draws the rectangle around it
    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return plate_img


# reading in the input image
plate = cv2.imread(image_path)

# need to change color of picture from BGR to RGB
plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
# display(plate, image_path.rsplit('.')[0]+'_diff_colors.'+image_path.rsplit('.')[1])

# Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier(haar_path)

result = detect_plate(plate)
display(result, image_path.rsplit('.')[0]+'_out.'+image_path.rsplit('.')[1])
