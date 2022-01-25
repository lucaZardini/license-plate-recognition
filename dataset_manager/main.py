import os

from dataset import DatasetSize


def main():
    dataset = DatasetSize.build(path=os.getenv("DATASET_PATH"), name=os.getenv("DATASET_NAME"), label_path=os.getenv("DATASET_LABEL"))

    dataset.separate_img_and_labels()
    # print(len(dataset.images))
    # print(dataset.images_are_squared)
    # print(len(dataset.format))
    # print(dataset.format[0])
    # if not dataset.images_are_squared:
    #     yolov5 takes as input squared images

    # if len(dataset.image_sizes) != 1:
    #     select a size and change the size of some images

    # if len(dataset.format) != 1:
    #     change all the images in jpeg

    # if dataset.format[0] != ".jpg" and dataset.format[0] != "jpeg":
    #     set all images to jpg or jpeg

    # print(len(dataset.image_sizes))
    # print(dataset.images[0].width)
    # print(dataset.images[0].height)

    # if dataset.image_sizes[0] != (int(os.getenv("DATASET_SIZE")), int(os.getenv("DATASET_SIZE"))):
    #     dataset.change_size(new_size=int(os.getenv("DATASET_SIZE")))



if __name__ == "__main__":
    main()