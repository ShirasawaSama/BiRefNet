import torch


weights_file = 'BiRefNet_HR-matting-epoch_135.pth'  # https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet_dynamic-general-epoch_174.pth
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic


@parse_args(
    "v",  # arg0: input (tensor)
    "v",  # arg1: weight (tensor)
    "v",  # arg2: offset (tensor)
    "v",  # arg3: mask (tensor)
    "v",  # arg4: bias (tensor)
    "i",  # arg5: stride_h
    "i",  # arg6: stride_w
    "i",  # arg7: pad_h
    "i",  # arg8: pad_w
    "i",  # arg9: dilation_h
    "i",  # arg10: dilation_w
    "i",  # arg11: groups
    "i",  # arg12: deform_groups
    "b",  # arg13: some bool
)
def symbolic_deform_conv_19(
    g,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    deform_groups,
    maybe_bool,
):
    strides = [stride_h, stride_w]
    pads = [pad_h, pad_w, pad_h, pad_w]
    dilations = [dilation_h, dilation_w]

    return g.op(
        "DeformConv",
        input,
        weight,
        offset,
        bias,
        mask,
        strides_i=strides,
        pads_i=pads,
        dilations_i=dilations,
        group_i=groups,
        offset_group_i=deform_groups,
        # You can ignore maybe_bool if you don't need it, or pass it as an attribute.
    )


register_custom_op_symbolic(
    "torchvision::deform_conv2d",  # PyTorch JIT/FX name
    symbolic_deform_conv_19,
    opset_version=19,
)

from utils import check_state_dict
from models.birefnet import BiRefNet


birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('./{}'.format(weights_file), map_location=device, weights_only=True)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

torch.set_float32_matmul_precision(['highest'][0])

birefnet.to(device)
_ = birefnet.eval()

def convert_to_onnx(net, file_name='output.onnx', input_shape=(2048, 2048), device=device):
    input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    input_layer_names = ['input_image']
    output_layer_names = ['output_image']

    torch.onnx.export(
        net,
        input,
        file_name,
        verbose=False,
        dynamo=False,
        opset_version=19,
        input_names=input_layer_names,
        output_names=output_layer_names,
        external_data=True,
    )
convert_to_onnx(birefnet, weights_file.replace('.pth', '.onnx'), input_shape=(1024, 1024), device=device)