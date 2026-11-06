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
    """Pré-processamento letterbox do Faveren; inferência ONNX FP32 CPU.

    O método raw() mede apenas session.run e materialização das saídas.
    A implementação de detect() fica pendente de validação da qualidade.
    """
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
        inp = self.session.get_inputs()[0]
        if inp.type != "tensor(float)":
            raise ValueError(f"{path.name}: esperado FP32, recebido {inp.type}")
        dims = inp.shape
        if len(dims) != 4 or (isinstance(dims[0], int) and dims[0] != 1) or dims[1] != 3:
            raise ValueError(f"{path.name}: esperado BCHW com batch 1 e 3 canais: {dims}")
        if not isinstance(dims[2], int) or not isinstance(dims[3], int):
            raise ValueError(f"{path.name}: altura/largura devem ser estáticas: {dims}")
        self.height, self.width = dims[2], dims[3]
        self.input_name = inp.name

    def prepare_path(self, path: Path) -> np.ndarray:
        from PIL import ImageOps
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        iw, ih = image.size
        scale = min(self.width / iw, self.height / ih)
        nw = min(self.width, round(iw * scale))
        nh = min(self.height, round(ih * scale))
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        canvas.paste(resized, ((self.width - nw) // 2, (self.height - nh) // 2))
        self._geometry = (iw, ih, nw, nh, (self.width-nw)//2, (self.height-nh)//2)
        arr = np.asarray(canvas, dtype=np.float32)
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None] / 255.0)

    def raw(self, value: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: value})

    def detect(self, value: np.ndarray, threshold: float):
        """YOLO Ultralytics 1 classe, saída [1,5,8400], xywh em pixels de letterbox.

        As caixas retornadas estão normalizadas nas coordenadas da imagem original.
        É obrigatório chamar prepare_path() antes para registrar a transformação.
        """
        if not hasattr(self, '_geometry'):
            raise RuntimeError('Use prepare_path(caminho) antes de detect() para desfazer letterbox')
        outputs = self.raw(value)
        pred = outputs[0]
        if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[1] != 5:
            raise ValueError(f'Saída YOLO inesperada: {pred.shape}; esperado [1,5,N] para uma classe')
        vals = pred[0].T
        valid = np.isfinite(vals).all(axis=1) & (vals[:,4] >= threshold)
        vals = vals[valid]
        if not len(vals):
            return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        xywh = vals[:, :4]
        scores = vals[:, 4].astype(np.float32)
        boxes = np.column_stack((xywh[:,0]-xywh[:,2]/2, xywh[:,1]-xywh[:,3]/2,
                                 xywh[:,0]+xywh[:,2]/2, xywh[:,1]+xywh[:,3]/2))
        keep = nms(boxes, scores, threshold=0.6, limit=100)
        boxes, scores = boxes[keep], scores[keep]
        iw, ih, nw, nh, padx, pady = self._geometry
        boxes[:, [0,2]] = (boxes[:, [0,2]] - padx) * (iw/nw) / iw
        boxes[:, [1,3]] = (boxes[:, [1,3]] - pady) * (ih/nh) / ih
        boxes = np.clip(boxes, 0.0, 1.0)
        valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
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
