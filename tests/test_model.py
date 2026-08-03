from pathlib import Path
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)

print("\nINPUTS:")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nOUTPUTS:")
for out in session.get_outputs():
    print(out.name, out.shape, out.type)
