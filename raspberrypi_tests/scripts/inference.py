from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 640


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.6, limit: int = 100) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        area_current = np.maximum(0, boxes[current, 2] - boxes[current, 0]) * np.maximum(0, boxes[current, 3] - boxes[current, 1])
        area_rest = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0, boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-12)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


class TFLiteDetector:
    layout = "nhwc"

    def __init__(self, path: Path, threads: int):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow import lite
            Interpreter = lite.Interpreter
        self.interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = {detail["name"].rsplit(":", 1)[-1]: detail for detail in self.interpreter.get_output_details()}

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        # export_tflite_graph_tf2 espera pixels float32 normalizados em [0, 1].
        return (rgb / 255.0)[None].astype(self.input["dtype"], copy=False)

    def raw(self, value: np.ndarray) -> list[np.ndarray]:
        self.interpreter.set_tensor(self.input["index"], value)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(item["index"]) for item in self.interpreter.get_output_details()]

    def detect(self, value: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        self.interpreter.set_tensor(self.input["index"], value)
        self.interpreter.invoke()
        values = {suffix: self.interpreter.get_tensor(detail["index"]) for suffix, detail in self.outputs.items()}
        count = int(values["0"][0])
        scores = values["1"][0, :count]
        ymin_xmin_ymax_xmax = values["3"][0, :count]
        boxes = ymin_xmin_ymax_xmax[:, [1, 0, 3, 2]]
        selected = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= threshold)
        return np.clip(boxes[selected], 0.0, 1.0), scores[selected]


class NanoDetONNXDetector:
    layout = "nchw"
    strides = (8, 16, 32, 64)
    reg_max = 7

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        priors = []
        for stride in self.strides:
            size = IMAGE_SIZE // stride
            y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            priors.append(np.stack((x.reshape(-1), y.reshape(-1)), axis=1) * stride)
        self.centers = np.concatenate(priors).astype(np.float32)
        self.stride_values = np.concatenate([
            np.full((IMAGE_SIZE // stride) ** 2, stride, dtype=np.float32) for stride in self.strides
        ])

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        bgr = rgb[..., ::-1]
        normalized = (bgr - np.array([103.53, 116.28, 123.675], dtype=np.float32)) / np.array(
            [57.375, 57.12, 58.395], dtype=np.float32
        )
        return np.transpose(normalized, (2, 0, 1))[None].astype(np.float32, copy=False)

    def raw(self, value: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: value})

    def detect(self, value: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        output = self.raw(value)[0][0]
        scores = output[:, 0]
        selected = scores >= threshold
        scores = scores[selected]
        distributions = output[selected, 1:].reshape(-1, 4, self.reg_max + 1)
        distributions -= distributions.max(axis=2, keepdims=True)
        probabilities = np.exp(distributions)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * np.arange(self.reg_max + 1, dtype=np.float32)).sum(axis=2)
        distances *= self.stride_values[selected, None]
        centers = self.centers[selected]
        boxes = np.column_stack((
            centers[:, 0] - distances[:, 0], centers[:, 1] - distances[:, 1],
            centers[:, 0] + distances[:, 2], centers[:, 1] + distances[:, 3],
        ))
        keep = nms(boxes, scores, threshold=0.6, limit=100)
        return np.clip(boxes[keep] / IMAGE_SIZE, 0.0, 1.0), scores[keep]


class YOLOONNXDetector:
    """Ultralytics ONNX de uma classe, com letterbox e saída ``[1, 5, N]``."""

    layout = "nchw"

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        dimensions = model_input.shape
        if model_input.type != "tensor(float)":
            raise ValueError(f"{path.name}: esperado FP32, recebido {model_input.type}")
        if (len(dimensions) != 4 or dimensions[1] != 3
                or not isinstance(dimensions[2], int) or not isinstance(dimensions[3], int)):
            raise ValueError(f"{path.name}: esperado BCHW estático: {dimensions}")
        self.height, self.width = dimensions[2], dimensions[3]
        self.input_name = model_input.name

    def prepare_path(self, path: Path) -> np.ndarray:
        from PIL import ImageOps
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        image_width, image_height = image.size
        scale = min(self.width / image_width, self.height / image_height)
        resized_width = min(self.width, round(image_width * scale))
        resized_height = min(self.height, round(image_height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (self.width - resized_width) // 2
        pad_y = (self.height - resized_height) // 2
        canvas = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        self._geometry = (resized_width, resized_height, pad_x, pad_y)
        array = np.asarray(canvas, dtype=np.float32)
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None] / 255.0)

    def raw(self, value: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: value})

    def detect(self, value: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_geometry"):
            raise RuntimeError("Use prepare_path() antes de detect() para desfazer o letterbox.")
        prediction = self.raw(value)[0]
        if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[1] != 5:
            raise ValueError(f"Saída YOLO inesperada: {prediction.shape}; esperado [1, 5, N].")
        values = prediction[0].T
        values = values[np.isfinite(values).all(axis=1) & (values[:, 4] >= threshold)]
        if not len(values):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        xywh = values[:, :4]
        scores = values[:, 4].astype(np.float32)
        boxes = np.column_stack((
            xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2,
        ))
        keep = nms(boxes, scores, threshold=0.6, limit=100)
        boxes, scores = boxes[keep], scores[keep]
        resized_width, resized_height, pad_x, pad_y = self._geometry
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / resized_width
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / resized_height
        boxes = np.clip(boxes, 0.0, 1.0)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid].astype(np.float32), scores[valid]


def detector(spec: dict, root: Path, threads: int):
    path = root / spec["path"]
    if spec["format"] == "tflite":
        return TFLiteDetector(path, threads)
    if spec["format"] == "onnx" and spec["name"] in ("yolov5nu", "yolov8n", "yolo11n"):
        return YOLOONNXDetector(path, threads)
    if spec["format"] == "onnx":
        return NanoDetONNXDetector(path, threads)
    raise ValueError(f"Formato não suportado: {spec['format']}")
