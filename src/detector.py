from pathlib import Path
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"


class Detector:

    def __init__(self):

        options = ort.SessionOptions()

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, image):

        output = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )

        return output[0]
