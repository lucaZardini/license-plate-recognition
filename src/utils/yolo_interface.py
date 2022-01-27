from src.utils.constants import ROOT
import subprocess


class YoloInterface:

    WEIGHTS = ROOT / "yolov5" / "weights" / "yolov5n_custom.pt"

    @staticmethod
    def run_yolo_detect(source: str, destination: str, hide_logs: bool):
        subprocess.call(
            ['python', ROOT / 'yolov5/detect.py',
             '--weights', YoloInterface.WEIGHTS,
             '--source', source,
             '--name', destination],
            stdout=subprocess.DEVNULL if hide_logs else None,
        )
