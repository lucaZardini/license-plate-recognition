# License Plate Detection using image processing techniques
This project aims at detecting a license plate in an image using image processing techniques.

This project was developed for the *Signal, Image and Video* course of the Master Degree *Artificial Intelligence Systems* at the University of Trento.

Authors:
* [Nicola Farina](https://github.com/nicola-farina)
* [Luca Zardini](https://github.com/lucaZardini)

## Usage
1. Clone the repository:
```
git clone -b image_processing --single-branch https://github.com/lucaZardini/license-plate-recognition.git
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
4. Create an `images` folder and put the images to run detection on in that folder. If you want to use your own folder, you can specify it with the `--img_folder` flag when running the script. If you want to use a single file, use `--img_path` instead. 

   (*WARNING: paths are relative to the project root*)

5. The results are stored by default in `results` folder. You can specify a different folder using the `--destination` flag. 

   (*WARNING: paths are relative to the project root*)

6. Now, simply run the script (with any optional flag you need):
```
cd src
python main.py
```
