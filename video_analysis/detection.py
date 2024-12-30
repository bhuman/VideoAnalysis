from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

if TYPE_CHECKING:
    import os

    import numpy as np
    import numpy.typing as npt


class Detector:
    """Encapsulates YOLOv5's object detection.

    This class is based on code from YOLOv5's detect.py.
    """

    def __init__(
        self,
        weights: os.PathLike | str,
        imgsz: list[int] | None = None,
        device: str = "",
        dnn: bool = False,
        data: os.PathLike | str | None = None,
        half: bool = False,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: list[int] | None = None,
        agnostic_nms: bool = False,
        max_det: int = 11,
    ) -> None:
        """Initialize the detector.

        :param weight: The path the file containing the network weights.
        :param imgsz: The height and width of the input images (in that order).
        :param device: CUDA device, i.e. 0 or 0,1,2,3 or cpu.
        :param dnn: Use OpenCV DNN for ONNX inference.
        :param data: The path to the yaml-file that describes the dataset.
        :param half: Use FP16 half-precision inference.
        :param conf_thres: Minimum confidence accepted.
        :param iou_thres: Non-maximum suppression intersection-over-union threshold.
        :param classes: Only these classes can be detected.
        :param agnostic_nms: Use class-agnostic non-maximum suppression.
        :param max_det: Maximum number of detections per image.
        """
        self._conf_thres: float = conf_thres
        self._iou_thres: float = iou_thres
        self._classes = classes
        self._agnostic_nms: bool = agnostic_nms
        self._max_det: int = max_det

        with torch.no_grad():
            self._device: torch.device = select_device(device)
            self.model = DetectMultiBackend(
                weights,  # pyright: ignore[reportGeneralTypeIssues]
                device=self._device,
                dnn=dnn,
                data=data,
                fp16=half,
            )
            self.imgsz: list[int] = check_img_size(imgsz, s=self.model.stride)  # pyright: ignore[reportGeneralTypeIssues]
            assert isinstance(self.imgsz, list)

            self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def run(
        self, preprocessed_image: npt.NDArray[np.uint8], original_shape: tuple[int, ...] | None = None
    ) -> npt.NDArray[np.float_]:
        """Run the detector.

        :param preprocessed_image: The preprocessed input image for the network.
        :param original_shape: The original shape of the input image.
        :return: A list of detections in the format `[x_min, y_min, x_max, y_max, confidence, class]`.
        """
        # preprocessed_image must be a numpy array of shape [1x]CxHxW, where C=3 (RGB), H=imgsz[0] and W=imgsz[1].
        assert len(preprocessed_image.shape) == 3 or (
            len(preprocessed_image.shape) == 4 and preprocessed_image.shape[0] == 1
        )
        assert preprocessed_image.shape[-3] == 3
        assert preprocessed_image.shape[-2] == self.imgsz[0]
        assert preprocessed_image.shape[-1] == self.imgsz[1]

        with torch.no_grad():
            image = torch.from_numpy(preprocessed_image).to(self._device)
            image = image.half() if self.model.fp16 else image.float()  # uint8 to fp16/32
            image /= 255  # 0 - 255 to 0.0 - 1.0
            if len(image.shape) == 3:
                image = image[None]  # expand for batch dim

            # Inference
            pred = self.model(image)

            # NMS
            pred = non_max_suppression(
                pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms, max_det=self._max_det
            )

            # non_max_suppression creates a list with an element per batch item. There should be exactly one.
            assert len(pred) == 1
            det = pred[0]

            if original_shape is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], original_shape).round()

        return det.cpu().numpy()
