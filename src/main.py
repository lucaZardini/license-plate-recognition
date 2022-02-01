import argparse

from yolo_interface import YoloInterface
from utils.constants import ROOT


def main():

    arg_parser = argparse.ArgumentParser(description="Car plate recognition. Use either --validate or --detect")
    arg_parser.add_argument("--validate", action="store_true", required=False, help="Perform validation on a labeled dataset")
    arg_parser.add_argument("--detect", action="store_true", required=False, help="Perform detection on unlabeled images")
    arg_parser.add_argument("--img_path", type=str, required=False, help="The path of a single image for detection")
    arg_parser.add_argument("--img_folder", type=str, default="detection/images", required=False, help="The path of the folder that contains the images for detection. Default is 'detection/images'")
    arg_parser.add_argument("--destination", required=False, default="results", help="The path where to save the results. Default is '[validation/detection]/results'")
    arg_parser.add_argument("--weights", required=False, default="yolov5n_custom.pt", help="Name of weights file in yolov5/weights folder")
    args = arg_parser.parse_args()

    if args.validate:
        destination = args.destination if args.destination != "results" else "validation/results"
        YoloInterface.run_yolo_val(destination=ROOT/destination, weights=args.weights)
    elif args.detect:
        destination = args.destination if args.destination != "results" else "detection/results"
        if args.img_path:
            YoloInterface.run_yolo_detect(source=ROOT/args.img_path, destination=ROOT/destination, weights=args.weights)
        else:
            YoloInterface.run_yolo_detect(source=ROOT/args.img_folder, destination=ROOT/destination, weights=args.weights)
    else:
        print("ERROR: either use the flag --detect or --validate")


if __name__ == "__main__":
    main()
