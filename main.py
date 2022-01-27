import argparse

from src.utils.yolo_interface import YoloInterface
from src.utils.constants import ROOT


def main():

    arg_parser = argparse.ArgumentParser(description="Car plate recognition. By default, all images in /images are used for detection and results are stored in /result")
    arg_parser.add_argument("-i", "--img_path", type=str, required=False, help="The path of a single image")
    arg_parser.add_argument("-f", "--img_folder", type=str, default="images", required=False, help="The path of the folder that contains the images")
    arg_parser.add_argument("-d", "--destination", required=False, default="results", help="The path where to save the results")
    arg_parser.add_argument("-l", "--hide_logs", required=False, default=False, help="Hide logs if set")
    args = arg_parser.parse_args()

    if args.img_path:
        YoloInterface.run_yolo_detect(source=ROOT / args.img_path, destination=ROOT / args.destination, hide_logs=args.hide_logs)
    elif args.img_folder:
        YoloInterface.run_yolo_detect(source=ROOT / args.img_folder, destination=ROOT / args.destination, hide_logs=args.hide_logs)


if __name__ == "__main__":
    main()
