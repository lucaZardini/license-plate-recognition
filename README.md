# License Plate Detection using YOLOv5
This project is a wrapper around a custom-trained YOLOv5 model for license plate detection in images.

This project was developed for the *Signal, Image and Video* course of the Master Degree *Artificial Intelligence Systems* at the University of Trento.

Authors:
* [Nicola Farina](https://github.com/nicola-farina)
* [Luca Zardini](https://github.com/lucaZardini)

## Usage
This project includes a YOLOv5 model with custom-trained weights, that allows you to run either detection (inference on unlabeled images) or validation (inference on labeled images, with performance results).
1. Clone the repository:
```
git clone https://github.com/lucaZardini/car-plate-recognition.git
```
2. *(Recommended)* Create a Python virtual environment in the repository folder, and activate it:
```
cd license-plate-recognition
python3 -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
4. For **detection**:
   * Put the images you want to run detection on in the `detection/images` folder. We already put some images there for convenience.
    If you want, you can specify a custom path with `--img-folder YOUR_IMG_FOLDER`, or a single image with `--img-path YOUR_IMG`.
   * The annotated images will be stored in `detection/results`. If you want, you can specify a custom path with `--destination YOUR_DESTINATION_FOLDER`.
   * Run `python main.py --detect`, with any optional flags you need.

5. For **validation**:
   * We provided a small set of images and labels for convenience. They can be found in `validation/dataset`. If you want to use your own images and labels, replace the current ones by making sure to keep the same folder structure.
   * The results of validation will be stored in `validation/results`. If you want, you can specify a custom path with `--destination YOUR_DESTINATION_FOLDER`.
   * Run `python main.py --validate`, with any optional flags you need.
