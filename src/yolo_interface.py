from src.utils.constants import ROOT
import subprocess


class YoloInterface:

    WEIGHTS_PATH = ROOT / "yolov5" / "weights"

    @staticmethod
    def run_yolo_detect(source: str, destination: str, weights: str):
        subprocess.call(
            ['python', ROOT / 'yolov5/detect.py',
             '--weights', YoloInterface.WEIGHTS_PATH / weights,
             '--source', source,
             '--name', destination,
             '--conf-thres', '0.16']
        )

    @staticmethod
    def run_yolo_val(destination: str, weights: str):
        subprocess.call(
            ['python', ROOT / 'yolov5/val.py',
             '--weights', YoloInterface.WEIGHTS_PATH / weights,
             '--data', ROOT / "validation" / 'data.yaml',
             '--task', 'test',
             '--single-cls',
             '--project', destination,
             '--name', 'test']
        )
