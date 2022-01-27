from dataclasses import dataclass
from yolov5 import val

@dataclass
class YoloInterface:
    image_folder: str
    destination_folder: str

    def run_yolo_val(self):
        val.main(
            ['--weights', '',  # TODO
             '--data', 'json',
             '--task', 'test',
             '--name', '']  # TODO
        )
