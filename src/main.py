import argparse

from src.image_processing.canny import CannyEdge
from src.utils.constants import ROOT


def main():
    arg_parser = argparse.ArgumentParser(description="Car plate recognition with image processing method")
    arg_parser.add_argument("--img_path", type=str, required=False, help="The path of a single image for detection")
    arg_parser.add_argument("--img_folder", type=str, default="images", required=False,
                            help="The path of the folder that contains the images for detection. Default is 'images'")
    arg_parser.add_argument("--destination", required=False, default="results",
                            help="The path where to save the results. Default is 'results'")
    args = arg_parser.parse_args()

    if args.img_path:
        CannyEdge.run_detection(source=str(ROOT / args.img_path), destination=str(ROOT / args.destination), is_folder=False)
    else:
        CannyEdge.run_detection(source=str(ROOT / args.img_folder), destination=str(ROOT / args.destination), is_folder=True)


if __name__ == "__main__":
    main()
