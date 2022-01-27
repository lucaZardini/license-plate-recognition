from __future__ import annotations, absolute_import

import os
import shutil
from dataclasses import dataclass
from typing import List

import PIL


@dataclass
class Image:
    name: str
    path: str
    width: int
    height: int
    label_path: str

    @property
    def is_square(self) -> bool:
        return self.height == self.width


@dataclass
class Dataset:
    name: str
    path: str
    number_files: int
    formats: List[str]
    images: List[Image]

    @property
    def image_are_same_format(self) -> bool:
        return len(self.formats) == 1

    @property
    def images_are_square(self) -> bool:
        for image in self.images:
            if not image.is_square:
                return False
        return True

    @property
    def image_sizes(self) -> List[(int, int)]:
        sizes = []
        for image in self.images:
            if (image.width, image.height) not in sizes:
                sizes.append((image.width, image.height))
        return sizes

    def change_size(self, new_size: int, resized_folder: str) -> None:
        image_path = resized_folder+"/images"
        label_path = resized_folder+"/labels"
        if os.path.exists(resized_folder):
            if not os.path.exists(resized_folder+"/images"):
                os.makedirs(resized_folder+"/images")
            if not os.path.exists(resized_folder+"/labels"):
                os.makedirs(resized_folder+"/labels")
        else:
            raise IOError
        for imm in self.images:
            image = PIL.Image.open(imm.path)
            image.resize((new_size, new_size))
            image.save(image_path+"/"+imm.name)
            last_point = imm.name.rfind(".")
            label_name = imm.name[:last_point]+".txt"
            self._recalculate_labels(label_path=imm.label_path, new_size=new_size, old_width=imm.width, old_height=imm.height, labels_path=label_path, label_name=label_name)

    def separate_img_and_labels(self):
        if not os.path.exists(self.path+"/images"):
            os.makedirs(self.path+"/images")
        if not os.path.exists(self.path+"/labels"):
            os.makedirs(self.path+"/labels")
        for image in self.images:
            new_path = self.path+"/images/"+image.name
            last_point = image.name.rfind(".")
            label_name = image.name[:last_point]
            new_path_label = self.path+"/labels/"+label_name+".txt"
            shutil.move(image.path, new_path)
            shutil.move(image.label_path, new_path_label)

    @staticmethod
    def _recalculate_labels(label_path: str, new_size: int, old_width: int, old_height: int, labels_path: str, label_name: str) -> None:
        image_class, x, y, w, h = Dataset.load_label(label_path, old_width, old_height)
        with open(labels_path + "/" + label_name, "w") as f:

            for i in range(len(image_class)):
                new_x = x[i]*new_size/old_width
                new_y = y[i]*new_size/old_height
                new_w = w[i]*new_size/old_width
                new_h = h[i]*new_size/old_height
                new_s = ""+str(image_class[i])+" "+str(new_x/new_size)+" "+str(new_y/new_size)+" "+str(new_w/new_size)+" "+str(new_h/new_size)+"\n"
                f.write(new_s)

    @staticmethod
    def load_label(label_path: str, old_width: int, old_height: int) -> (List[int], List[float], List[float], List[float], List[float]):
        img_class = []
        x = []
        y = []
        w = []
        h = []
        number_lines = 0
        with open(label_path, "r") as f:
            for line in f:
                number_lines=number_lines+1
                values = line.split(" ")
                img_class.append(int(values[0]))
                x.append(float(values[1])*old_width)
                y.append(float(values[2])*old_height)
                w.append(float(values[3])*old_width)
                h.append(float(values[4])*old_height)
            return img_class, x, y, w, h

    @staticmethod
    def build(name: str, path: str, label_path: str) -> Dataset:
        files = os.listdir(path)
        images = []
        formats = []
        for file in files:
            if Dataset.file_is_image(file):
                filepath = path + "/" + file
                image = PIL.Image.open(filepath)
                width, height = image.size
                image_label_path = Dataset.label_path(label_path=label_path, image_name=file)
                images.append(
                    Image(name=file, path=filepath, height=height, width=width, label_path=image_label_path))
                last_point = file.rfind(".")
                file_format = file[last_point:]
                if file_format not in formats:
                    formats.append(file_format)
        return Dataset(name=name, path=path, number_files=len(images), formats=formats, images=images)

    @staticmethod
    def label_path(image_name: str, label_path: str) -> str:
        last_point = image_name.rfind(".")
        return label_path + "/" + image_name[:last_point] + ".txt"

    @staticmethod
    def file_is_image(filename: str):
        last_point = filename.rfind(".")
        return filename[last_point:] in [".png", ".jpeg", ".jpg"]
