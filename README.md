# License Plate Detection
This project aims at detecting a license plate.

This project was developed for the *Signal, Image and Video* course of the Master Degree *Artificial Intelligence Systems* at the University of Trento.

Authors:
* [Nicola Farina](https://github.com/nicola-farina)
* [Luca Zardini](https://github.com/lucaZardini)

## Usage
The project has two main branches:

* [**image-processing**](https://github.com/lucaZardini/license-plate-recognition/blob/image_processing/README.md)
* [**yolo**](https://github.com/lucaZardini/license-plate-recognition/blob/yolo/README.md)

They contain two different car plate recognition method implementations, the first using image processing and the second using the neural network YOLOv5. 

Switch to your branch of interest and follow the README instructions.

## Project Structure

### Image-processing
![img_process](https://user-images.githubusercontent.com/50495055/152204482-3466838e-156f-4e48-8675-5dfc7b0d2eac.png)

This project contains:
* **images**: folder that contains the images to detect
* **_README.md_**: contains the information to use the code
* **_requirements.txt_**: list of python libraries required to run the code
* **src**
  * **detector**: python package
    * **_detector.py_**: script that contains to code to detect license plates
  * **_main.py_**: script to run to start the application
  * **utils**: package containing useful code
    * **_constants.py_**: contains the constants used in project
    * **_images.py_**: contains all the image processing operations and utilities used  

### YOLOv5
![yolo](https://user-images.githubusercontent.com/50495055/152206670-0fff1b71-424b-4221-b4b8-1c1026d6e24d.png)

This project contains:
* **detection**
  * **images**: folder that contains the images to detect (no labels)
* **_README.md_**: contains the information to use the code
* **_requirements.txt_**: list of python libraries required to run the code
* **src**
  * **dataset_manager**: python package
    * **_dataset.py_**: contains code to organize dataset with some useful operation.
  * **_main.py_**: script to run to start the application
  * **utils**: package containing useful code
    * **_constants.py_**: contains the constants used in project (the root folder)
  * **_yolo_interface.py_**: script that runs the detect or validate yolov5 scripts
* **validation**
  * **dataset**: folder that contains the images for validation
  * **_data.yaml_**: yolov5 configuration file
* **yolov5**: folder that contains the clone of the yolov5 repository
