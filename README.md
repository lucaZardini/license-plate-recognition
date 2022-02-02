# License Plate Detection
This project aims at detecting a license plate.

This project was developed for the *Signal, Image and Video* course of the Master Degree *Artificial Intelligence Systems* at the University of Trento.

Authors:
* [Nicola Farina](https://github.com/nicola-farina)
* [Luca Zardini](https://github.com/lucaZardini)


The project has two main branches:
* **image-processing**
* **yolo**

## Usage
The two branches contain two different car plate recognition method implementations, one using image processing and the other using yolov5. Move in the two branches to use the two methods and follow the README instruction of the branches.

## Project Structure

### Image-processing
![img_process](https://user-images.githubusercontent.com/71773192/152195417-b657d71a-e847-499a-883c-bd50381e6182.png)

This project contains:
* images: folder that contains the images to detect
* README.md: contains the information to use the code
* requirements.txt: list of python libraries required to run correctly the code
* src
  * detector: python package
    * detector.py: script that contains to code to detect license plates
  * main.py: script to run to start the application
  * utils: package containing useful code
    * constants.py: contains the constants used in project (the root folder)
    * images.py: contains all the image processing operations used  

### Yolo
![yol](https://user-images.githubusercontent.com/71773192/152195453-e39a30ae-2675-45f8-a9d1-34199216230b.png)

This project contains:
* detection
  * images: folder that contains the images to detect
* README.md: contains the information to use the code
* requirements.txt: list of python libraries required to run correctly the code
* src
  * dataset_manager: python package
    * dataset.py: contains code to organize dataset with some useful operation.
  * main.py: script to run to start the application
  * utils: package containing useful code
    * constants.py: contains the constants used in project (the root folder)
  * yolo_interface.py: script that runs the detect or validate yolov5 scripts
  * yolov5: folder that contains the clone of the yolov5 repository
