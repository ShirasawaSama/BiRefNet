from PIL import Image
from torchvision import transforms

import onnxruntime
import matplotlib.pyplot as plt


transform_image = transforms.Compose([
    transforms.Resize((2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

imagepath = 'images_todo/0-onnx_test-2.jpg'
image = Image.open(imagepath)
image = image.convert("RGB") if image.mode != "RGB" else image
input_images = transform_image(image).unsqueeze(0).to('cpu')
input_images_numpy = input_images.cpu().numpy()

# weights_file = 'BiRefNet_HR-matting-epoch_135.pth'


providers = ['DmlExecutionProvider']
run_options = onnxruntime.SessionOptions()
run_options.enable_profiling = True
run_options.log_severity_level = 0
onnx_session = onnxruntime.InferenceSession(
    'BiRefNet_HR-matting-epoch_135_fp16.onnx', run_options,
    providers=providers
)
input_name = onnx_session.get_inputs()[0].name


from time import time
import torch

time_st = time()
pred_onnx = torch.tensor(
    onnx_session.run(None, {input_name: input_images_numpy})[-1]
).squeeze(0).sigmoid().cpu()
print(time() - time_st)

plt.imshow(pred_onnx.squeeze(), cmap='gray'); plt.show()
