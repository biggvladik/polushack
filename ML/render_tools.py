from yolov5.models.common import Detections

class CustomDetections(Detections):
    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=...):
        return super()._run(pprint, show, save, crop, render, labels, save_dir)