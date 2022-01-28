import argparse

from src.yolo_interface import YoloInterface
from src.utils.constants import ROOT


def main():

    arg_parser = argparse.ArgumentParser(description="Car plate recognition. By default, all images in /images are used for detection and results are stored in /result")
    arg_parser.add_argument("-i", "--img_path", type=str, required=False, help="The path of a single image")
    arg_parser.add_argument("-f", "--img_folder", type=str, default="images", required=False, help="The path of the folder that contains the images")
    arg_parser.add_argument("-d", "--destination", required=False, default="results", help="The path where to save the results")
    arg_parser.add_argument("-w", "--weights", required=False, default="yolov5n_custom.pt", help="Name of weights file in yolov5/weights folder")
    args = arg_parser.parse_args()

    if args.img_path:
        YoloInterface.run_yolo_detect(source=ROOT / args.img_path, destination=ROOT / args.destination, weights=args.weights)
    elif args.img_folder:
        YoloInterface.run_yolo_detect(source=ROOT / args.img_folder, destination=ROOT / args.destination, weights=args.weights)


if __name__ == "__main__":
    main()
