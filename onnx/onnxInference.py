from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
from config import get_opts, get_training_size
from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3
import datasets.custom_transforms as custom_transforms
from visualization import *
import torch.onnx
import onnxruntime
import os
import time

def inference_onnx_model(img):
    ort_session = onnxruntime.InferenceSession('onnx/sc_depthv3.onnx')
    # numpy iput
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]
    out = torch.from_numpy(ort_out).float()
    return out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

## Inference
hparams = get_opts()

if hparams.model_version == 'v1':
    system = SC_Depth(hparams)
elif hparams.model_version == 'v2':
    system = SC_DepthV2(hparams)
elif hparams.model_version == 'v3':
    system = SC_DepthV3(hparams)

# training size
training_size = get_training_size(hparams.dataset_name)

# normalization
inference_transform = custom_transforms.Compose([
    custom_transforms.RescaleTo(training_size),
    custom_transforms.ArrayToTensor(),
    custom_transforms.Normalize()]
)

input_dir = Path(hparams.input_dir)
output_dir = Path(hparams.output_dir) / \
    'model_{}'.format(hparams.model_version)
output_dir.makedirs_p()

if hparams.save_vis:
    (output_dir/'vis').makedirs_p()

if hparams.save_depth:
    (output_dir/'depth').makedirs_p()

image_files = sum([(input_dir).files('*.{}'.format(ext))
                  for ext in ['jpg', 'png']], [])
image_files = sorted(image_files)

print('{} images for inference'.format(len(image_files)))

start_time = time.time() #增加计时器
for i, img_file in enumerate(tqdm(image_files)):

    filename = os.path.splitext(os.path.basename(img_file))[0]

    img = imread(img_file).astype(np.float32)
    tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
    pred_depth = inference_onnx_model(tensor_img)

    if hparams.save_vis:
        vis = visualize_depth(pred_depth[0, 0]).permute(
            1, 2, 0).numpy() * 255
        imwrite(output_dir/'vis/{}.jpg'.format(filename),
                vis.astype(np.uint8))

    if hparams.save_depth:
        depth = pred_depth[0, 0].cpu().numpy()
        np.save(output_dir/'depth/{}.npy'.format(filename), depth)

end_time = time.time()  # 记录结束时间
inference_time = (end_time - start_time) * 1000  # 推断时间（ms）
print("Inference time: %.2f ms" % inference_time)