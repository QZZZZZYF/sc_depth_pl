from config import get_opts, get_training_size
from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3
from imageio import imread, imwrite
import torch.onnx
import onnx
import onnxruntime
import time
import datasets.custom_transforms as custom_transforms
import numpy as np

# ## TO ONNX
hparams = get_opts()
if hparams.model_version == 'v1':
    system = SC_Depth(hparams)
elif hparams.model_version == 'v2':
    system = SC_DepthV2(hparams)
elif hparams.model_version == 'v3':
    system = SC_DepthV3(hparams)

system = system.load_from_checkpoint('/media/enb/d246805e-89c3-40bf-95da-9f0d81ea7b05/home/enb/ZYF/sc_depth_pl-master/ckpts/nyu_scv3/epoch=93-val_loss=0.1384.ckpt', strict=False)

model = system.depth_net
model.cuda()
model.eval()
# 模型输入的维度
image = torch.randn(1, 3, 480, 640).cuda()
onnx_model_path = 'sc_depthv3.824.onnx'

torch.onnx.export(
    model,  # 转换的模型
    image,  # 输入的维度
    onnx_model_path,  # 导出的 ONNX 文件名
    export_params=True,  # store the trained parameter weights inside the model file
    # verbose=True,
    opset_version=11,  # ONNX 算子集的版本
    input_names=["image"],  # 输入的 tensor名称，可变
    output_names=["depth"]  # 输出的 tensor名称，可变
    )
print("sucessful")

## check onnx
onnx_model = onnx.load("sc_depthv3.824.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

######################## Inference_img_dir
# def preprocess_image(image_path):
#     # training size
#     training_size = get_training_size(hparams.dataset_name)
#
#     # normalization
#     inference_transform = custom_transforms.Compose([
#         custom_transforms.RescaleTo(training_size),
#         custom_transforms.ArrayToTensor(),
#         custom_transforms.Normalize()]
#     )
#     img = imread(image_path).astype(np.float32)
#     tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
#     return tensor_img
#
# def inference_onnx_model(img):
#     ort_session = onnxruntime.InferenceSession('sc_depthv3.824.onnx')
#     # numpy iput
#     ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
#     ort_outs = ort_session.run(None, ort_inputs)
#     ort_out = ort_outs[0]
#     out = torch.from_numpy(ort_out).float()
#     return out
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# def show_depth_image(out, out_flip):
#
#     return out
#
# img = 'demo/time/input/1.jpg'
# ort_session = onnxruntime.InferenceSession('sc_depthv3.824.onnx')
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
# t0 = time.time()
# for i in range(1000):
#     ort_outs = ort_session.run(None, ort_inputs)
# t1 = time.time()
# print("用onnx完成1000次推理消耗的时间:%s" % (t1-t0))
