from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import check_file


class SourceAdapter:
    """Provider for video frames.

    This class is loosely based on code from YOLOv5's detect.py.
    """

    def __init__(self, source: str, imgsz: list[int] | int, stride: int, pt: bool, step: int = 1) -> None:
        """Initialize this provider.

        :param source: Path or URL of the video or a folder with images. Could also be the
        number of the webcam (which is currently not supported).
        :param imgsz: The height and width of the image (in that order) or a single number
        if both sizes are the same.
        :param stride: Stride of the model. Image size is increased to a multiple of this value.
        :param pt: Using PyTorch?
        :param step: In which steps is moved through the video? 1 returns every frame, 2 every
        second frame, etc.
        """
        is_file: bool = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url: bool = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam: bool = source.isnumeric() or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)

        if self.webcam:
            self._dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)  # type: ignore[arg-type]
            assert len(self._dataset) == 1
            self.fps = self._dataset.fps[0]
        else:
            self._dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # type: ignore[arg-type]
            assert self._dataset.cap is not None
            self.fps = self._dataset.cap.get(cv2.CAP_PROP_FPS)
        self.fps /= step

        self._step = step
        self._iter = None
        self._skip: int = 0

    def frames(self) -> int:
        """Return the overall number of frames if known.

        :return: The number of frames that will be played back or 0 if this is a stream.
        """
        return self._dataset.frames // self._step if isinstance(self._dataset.frames, int) else 0

    def __iter__(self) -> SourceAdapter:
        """Return an iterator for image pairs.

        Since this object is also used as its iterator, only a single iterator
        can be used at a time. Multithreading is also not supported.
        """
        self._iter = self._dataset.__iter__()
        self._skip = 0
        return self

    def __next__(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Returns the next pair of preprocessed and original image.

        :return: The image pair. The first one is preprocessed to be usable as YOLOv5 input.
        """
        assert self._iter is not None
        for _ in range(self._skip):
            self._iter.__next__()
        path, preprocessed, original, _, _ = self._iter.__next__()
        if self.webcam:
            assert len(path) == 1
            assert preprocessed.shape[0] == 1
            assert len(original) == 1
            preprocessed = preprocessed[0, ...]
            original = original[0]
        self._skip = self._step - 1
        return preprocessed, original  # pyright: ignore # MAT == NDArray[uint8]
