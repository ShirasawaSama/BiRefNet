"""
运行前先激活 venv:  source .venv/bin/activate  (或  source venv/bin/activate)

关于警告 "Some nodes were not assigned to the preferred execution providers":
  未分配到首选 EP（即被放在 CPU 上）的节点主要是两类：
  1) Reshape — ORT 会把 shape 相关算子显式放在 CPU 上以提升整体性能；
  2) MemcpyToHost / MemcpyFromHost — 数据搬运必须在 CPU 上。
  这是预期行为，一般不影响正确性；若不想看到该警告，可设 so.log_severity_level = 3。
"""
import onnxruntime

device = 'cuda'
weights_file = 'BiRefNet_dynamic-general-epoch_174.pth'  # https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet_dynamic-general-epoch_174.pth


providers = ['CPUExecutionProvider'] if device == 'cpu' else [("CUDAExecutionProvider")]

so = onnxruntime.SessionOptions()
# 0=Verbose(全部), 1=Info, 2=Warning, 3=Error, 4=Fatal
# 设为 2 或 3 可避免大量节点分配日志；设为 3 可连 "Some nodes were not assigned" 也不打印
# so.log_severity_level = 1
# so.log_verbosity_level = 1
# so.enable_profiling = True

onnx_session = onnxruntime.InferenceSession(
    weights_file.replace('.pth', '.onnx'),
    so,
    providers=providers
)
input_name = onnx_session.get_inputs()[0].name
print(onnxruntime.get_device(), onnx_session.get_providers())

from PIL import Image
import torch
from torchvision import transforms
from time import time

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

imagepath = 'image.png'
image = Image.open(imagepath)
image = image.convert("RGB") if image.mode != "RGB" else image
input_images = transform_image(image).unsqueeze(0).to(device)
input_images_numpy = input_images.cpu().numpy()
pred_onnx = torch.tensor(
    onnx_session.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
).squeeze(0).sigmoid().cpu()
pred_onnx = torch.tensor(
    onnx_session.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
).squeeze(0).sigmoid().cpu()

time_st = time()
pred_onnx = torch.tensor(
    onnx_session.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
).squeeze(0).sigmoid().cpu()
print(time() - time_st)