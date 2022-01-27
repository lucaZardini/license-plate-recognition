from utils.constants import ROOT
from yolov5 import detect


class YoloInterface:

    WEIGHTS = ROOT / "yolov5" / "weights" / "yolov5n_custom.pt"

    @staticmethod
    def run_yolo_detect(source: str, destination: str):
        detect.main(
            ['--weights', YoloInterface.WEIGHTS,
             '--source', source,
             '--name', destination]
        )
