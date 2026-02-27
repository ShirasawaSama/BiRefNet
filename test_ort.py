import torch
import torchvision
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import os
import random
import time

# Global Configuration
USE_CUDA = True   # Set to True to enable CUDA
USE_FP16 = False  # Set to True to enable FP16 (Half Precision)
NUM_TESTS = 300   # Number of random test cases
DISABLE_TF32 = True # Disable TF32 for better precision comparison

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Disable TF32 if requested
if DISABLE_TF32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("TF32 disabled for higher precision.")

def get_output_shape(height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w):
    out_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    return int(out_h), int(out_w)

def run_single_test(case_id, params):
    batch_size = params['batch_size']
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    kernel_h = params['kernel_h']
    kernel_w = params['kernel_w']
    height = params['height']
    width = params['width']
    stride_h = params['stride_h']
    stride_w = params['stride_w']
    pad_h = params['pad_h']
    pad_w = params['pad_w']
    dilation_h = params['dilation_h']
    dilation_w = params['dilation_w']
    groups = params['groups']
    offset_groups = params['offset_groups']

    # Determine device and dtype
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if USE_FP16 else torch.float32
    onnx_dtype = TensorProto.FLOAT16 if USE_FP16 else TensorProto.FLOAT
    
    # Calculate output dimensions
    out_h, out_w = get_output_shape(height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w)
    
    if out_h <= 0 or out_w <= 0:
        return "SKIPPED (Invalid Output Shape)", 0, 0

    # Generate random data
    try:
        x = torch.randn(batch_size, in_channels, height, width, dtype=dtype).to(device)
        
        weight = torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w, dtype=dtype).to(device)
        
        offset_channels = offset_groups * 2 * kernel_h * kernel_w
        offset = torch.randn(batch_size, offset_channels, out_h, out_w, dtype=dtype).to(device)
        
        mask_channels = offset_groups * kernel_h * kernel_w
        mask = torch.rand(batch_size, mask_channels, out_h, out_w, dtype=dtype).to(device)
        
        bias = torch.randn(out_channels, dtype=dtype).to(device)

        # 1. Run PyTorch
        if USE_CUDA and torch.cuda.is_available():
            torch.cuda.synchronize()
        
				# Run for JIT
        torch_out = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=weight,
            bias=bias,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dilation_h, dilation_w),
            mask=mask
        )

        start_time = time.time()
        torch_out = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=weight,
            bias=bias,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dilation_h, dilation_w),
            mask=mask
        )
        if USE_CUDA and torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_time = (time.time() - start_time) * 1000 # ms

    except Exception as e:
        return f"SKIPPED (PyTorch Error: {str(e)})", 0, 0

    # 2. Build ONNX model
    onnx_model_path = f"temp_test_{case_id}.onnx"
    try:
        X_info = helper.make_tensor_value_info('X', onnx_dtype, [batch_size, in_channels, height, width])
        W_info = helper.make_tensor_value_info('W', onnx_dtype, [out_channels, in_channels // groups, kernel_h, kernel_w])
        Offset_info = helper.make_tensor_value_info('offset', onnx_dtype, [batch_size, offset_channels, out_h, out_w])
        Bias_info = helper.make_tensor_value_info('B', onnx_dtype, [out_channels])
        Mask_info = helper.make_tensor_value_info('mask', onnx_dtype, [batch_size, mask_channels, out_h, out_w])
        Y_info = helper.make_tensor_value_info('Y', onnx_dtype, [batch_size, out_channels, out_h, out_w])

        node_def = helper.make_node(
            'DeformConv',
            inputs=['X', 'W', 'offset', 'B', 'mask'],
            outputs=['Y'],
            strides=[stride_h, stride_w],
            pads=[pad_h, pad_w, pad_h, pad_w],
            dilations=[dilation_h, dilation_w],
            group=groups,
            offset_group=offset_groups,
            kernel_shape=[kernel_h, kernel_w],
        )

        graph_def = helper.make_graph(
            [node_def],
            'test_deform_conv_graph',
            [X_info, W_info, Offset_info, Bias_info, Mask_info],
            [Y_info]
        )

        model_def = helper.make_model(graph_def, producer_name='test_ort')
        model_def.opset_import[0].version = 19
        
        onnx.save(model_def, onnx_model_path)
    except Exception as e:
        return f"ERROR (ONNX Build: {str(e)})", 0, 0

    # 3. Run ONNX Runtime
    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_CUDA else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)
        
        ort_inputs = {
            'X': x.detach().cpu().numpy(),
            'W': weight.detach().cpu().numpy(),
            'offset': offset.detach().cpu().numpy(),
            'B': bias.detach().cpu().numpy(),
            'mask': mask.detach().cpu().numpy()
        }
        
				# Run for JIT
        ort_outs = session.run(['Y'], ort_inputs)

        start_time = time.time()
        ort_outs = session.run(['Y'], ort_inputs)
        ort_time = (time.time() - start_time) * 1000 # ms
        
        ort_out = ort_outs[0]
    except Exception as e:
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)
        return f"ERROR (ORT Run: {str(e)})", 0, 0
    
    if os.path.exists(onnx_model_path):
        os.remove(onnx_model_path)

    # 4. Compare
    try:
        # Move torch output to cpu for comparison
        torch_out_np = torch_out.detach().cpu().numpy()
        
        # Adjust tolerances
        rtol = 1e-2
        atol = 1e-2
  
        np.testing.assert_allclose(torch_out_np, ort_out, rtol=rtol, atol=atol)
        return "SUCCESS", torch_time, ort_time
    except AssertionError as e:
        torch_out_np = torch_out.detach().cpu().numpy()
        diff = np.abs(torch_out_np - ort_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Calculate value range for context
        max_val = np.max(np.abs(torch_out_np))
        mean_val = np.mean(np.abs(torch_out_np))
        
        # Calculate relative error
        # Avoid division by zero
        rel_diff = diff / (np.abs(torch_out_np) + 1e-7)
        max_rel_diff = np.max(rel_diff)
        
        return f"FAILURE (Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}, Val range: [0, {max_val:.2f}], Max Rel diff: {max_rel_diff:.4f})", torch_time, ort_time

def generate_random_params():
    groups = random.choice([1, 2, 4])
    
    in_channels_base = random.randint(1, 8)
    in_channels = in_channels_base * groups
    
    out_channels_base = random.randint(1, 8)
    out_channels = out_channels_base * groups
    
    kernel_h = random.randint(1, 5)
    kernel_w = random.randint(1, 5)
    
    stride_h = random.randint(1, 3)
    stride_w = random.randint(1, 3)
    
    pad_h = random.randint(0, 2)
    pad_w = random.randint(0, 2)
    
    dilation_h = random.randint(1, 3)
    dilation_w = random.randint(1, 3)
    
    min_h = (kernel_h - 1) * dilation_h + 1
    min_w = (kernel_w - 1) * dilation_w + 1
    
    height = random.randint(min_h, min_h + 32)
    width = random.randint(min_w, min_w + 32)
    
    batch_size = random.randint(1, 4)
    offset_groups = random.choice([1, 2])
    
    return {
        'batch_size': batch_size,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'height': height,
        'width': width,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'dilation_h': dilation_h,
        'dilation_w': dilation_w,
        'groups': groups,
        'offset_groups': offset_groups
    }

def main():
    print(f"Running with USE_CUDA={USE_CUDA}, USE_FP16={USE_FP16}, DISABLE_TF32={DISABLE_TF32}")
    if USE_CUDA:
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: USE_CUDA is True but PyTorch CUDA is not available. Will fall back to CPU.")
    
    results = {
        "SUCCESS": 0,
        "FAILURE": 0,
        "ERROR": 0,
        "SKIPPED": 0
    }
    
    total_torch_time = 0
    total_ort_time = 0
    valid_time_count = 0
    
    failures = []
    errors = []

    print(f"Starting {NUM_TESTS} random tests...")
    
    for i in range(NUM_TESTS):
        params = generate_random_params()
        result, t_torch, t_ort = run_single_test(i, params)
        
        status = result.split()[0]
        if status in results:
            results[status] += 1
        else:
            results["ERROR"] += 1
            
        if status == "SUCCESS":
            total_torch_time += t_torch
            total_ort_time += t_ort
            valid_time_count += 1
            
        if status == "FAILURE":
            failures.append((i, params, result))
            print(f"Test {i+1}/{NUM_TESTS}: {result}")
        elif status == "ERROR":
            errors.append((i, params, result))
            print(f"Test {i+1}/{NUM_TESTS}: {result}")
        elif (i + 1) % 50 == 0:
            print(f"Test {i+1}/{NUM_TESTS}: {result}")

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {NUM_TESTS}")
    print(f"Success: {results['SUCCESS']}")
    print(f"Failure: {results['FAILURE']}")
    print(f"Error:   {results['ERROR']}")
    print(f"Skipped: {results['SKIPPED']}")
    
    if valid_time_count > 0:
        avg_torch = total_torch_time / valid_time_count
        avg_ort = total_ort_time / valid_time_count
        print("\nPERFORMANCE (Average per successful run)")
        print(f"PyTorch: {avg_torch:.4f} ms")
        print(f"ORT:     {avg_ort:.4f} ms")
        if avg_ort > 0:
            print(f"Ratio (Torch/ORT): {avg_torch/avg_ort:.2f}x")
    
    if failures:
        print("\nFailures Details (First 5):")
        for i, params, res in failures[:5]:
            print(f"Case {i}: {res}")
            print(f"  Params: {params}")

    if errors:
        print("\nErrors Details (First 5):")
        for i, params, res in errors[:5]:
            print(f"Case {i}: {res}")
            print(f"  Params: {params}")

if __name__ == "__main__":
    main()
