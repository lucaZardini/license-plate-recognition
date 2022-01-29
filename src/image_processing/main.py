import argparse

from image_processing.canny import CannyEdge
from utils.constants import ROOT


def main():
    arg_parser = argparse.ArgumentParser(description="Car plate recognition with image processing method")
    arg_parser.add_argument("--img_path", type=str, required=False, help="The path of a single image for detection")
    arg_parser.add_argument("--img_folder", type=str, default="detection/images", required=False,
                            help="The path of the folder that contains the images for detection. Default is 'detection/images'")
    arg_parser.add_argument("--destination", required=False, default="results",
                            help="The path where to save the results. Default is '[validation/detection]/results'")
    args = arg_parser.parse_args()

    if args.img_path:
        CannyEdge.run_detection(source=args.img_path, destination=ROOT / args.destination, is_folder=False)
    else:
        CannyEdge.run_detection(source=args.img_folder, destination=ROOT / args.destination, is_folder=True)


if __name__ == "__main__":
    main()
