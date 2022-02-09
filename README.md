# License Plate Detection using YOLOv5
This project is a wrapper around a custom-trained YOLOv5 model for license plate detection in images.

This project was developed for the *Signal, Image and Video* course of the Master Degree *Artificial Intelligence Systems* at the University of Trento.

Authors:
* [Nicola Farina](https://github.com/nicola-farina)
* [Luca Zardini](https://github.com/lucaZardini)

## Usage
This project includes a YOLOv5 model with custom-trained weights, that allows you to run either detection (inference on unlabeled images) or validation (inference on labeled images, with performance results).
1. Clone this branch of the repository:
```
git clone -b yolo --single-branch https://github.com/lucaZardini/license-plate-recognition.git
```
2. *(Recommended)* Create a Python virtual environment in the repository folder, and activate it:
```
cd license-plate-recognition
python3 -m venv venv
source venv/bin/activate
```
3. Install the required packages (this may take a while due to the large size of Torch package, ~2GB):
```
pip install -r requirements.txt
```
4. For **detection**:
   * Create a `detection/images` folder and put the images you want to run detection on in that folder.
    If you want, you can specify a custom path with `--img-folder YOUR_IMG_FOLDER`, or a single image with `--img-path YOUR_IMG`.
    
    (*WARNING: paths are relative to the project root*)
    
   * The annotated images will be stored in `detection/results`. If you want, you can specify a custom path with `--destination YOUR_DESTINATION_FOLDER`.

    (*WARNING: paths are relative to the project root*)
    
   * Run (with any optional flags you need):
    ```
    cd src
    python main.py --detect
    ```

5. For **validation**:
   * Create a `dataset` folder in the folder `validation`. Create two folders `images` and `labels`, and inside of both of them create a `test` folder. In `images/test` put your images, in `labels/test` put your labels. This is the usual YOLOv5 dataset structure.
    
   * The results of validation will be stored in `validation/results`. If you want, you can specify a custom path with `--destination YOUR_DESTINATION_FOLDER`.

     (*WARNING: paths are relative to the project root*)
    
   * Run (with any optional flags you need): 
    ```
    cd src
    python main.py --validate
    ```
