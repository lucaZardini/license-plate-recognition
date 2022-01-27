import argparse


def main():

    arg_parser = argparse.ArgumentParser(description="Car plate recognition: if nothing is set, default is /images for the images and /result for the image with labels")
    arg_parser.add_argument("-i", "--img_path", type=str, required=False, help="The path of the image")
    arg_parser.add_argument("-f", "--img_folder", type=str, required=False, help="The path of the folder that contains the images")
    arg_parser.add_argument("-d", "--destination", required=False, help="The path to save the image with labels")
    args = arg_parser.parse_args()

    if args.img_path:
        # pass the image path to the recognizer
    elif args.img_folder:
        # pass the image folder to the recognizer
    if args.destination:
        # pass the destination path to the recognizer


if __name__ == "__main__":
    main()