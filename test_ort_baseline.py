import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import os
import random
import time

# ==========================================
# Configuration
# ==========================================
USE_CUDA = True       # Enable CUDA
USE_FP16 = True      # Default to FP32 for baseline check
DISABLE_TF32 = True   # Disable TF32
NUM_TESTS = 20        # Number of tests per op

# Tolerances for FP32
ATOL = 0
RTOL = 0

# ==========================================

# Set seeds
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

if DISABLE_TF32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("TF32 disabled for strict comparison.")

def run_grid_sample_test(case_id):
    # GridSample Test
    # Input: (N, C, H_in, W_in)
    # Grid: (N, H_out, W_out, 2)
    
    N = random.randint(1, 4)
    C = random.randint(1, 8)
    H_in = random.randint(16, 32)
    W_in = random.randint(16, 32)
    H_out = random.randint(16, 32)
    W_out = random.randint(16, 32)
    
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if USE_FP16 else torch.float32
    onnx_dtype = TensorProto.FLOAT16 if USE_FP16 else TensorProto.FLOAT
    
    # Scale inputs
    x = torch.randn(N, C, H_in, W_in, dtype=dtype, device=device)
    # Grid values in range [-1, 1] usually, but let's go slightly outside to test padding
    grid = torch.rand(N, H_out, W_out, 2, dtype=dtype, device=device) * 2.5 - 1.25
    
    # PyTorch
    if USE_CUDA: torch.cuda.synchronize()
    # align_corners=False is often default in newer ops or specific to ONNX version, 
    # ONNX GridSample attributes: align_corners (int), mode (string), padding_mode (string)
    # PyTorch defaults: mode='bilinear', padding_mode='zeros', align_corners=False (since 1.3 user needs to specify, but usually False matches simplified logic)
    # Let's test align_corners=0 (False) which is standard for modern resize/grid_sample
    torch_out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    if USE_CUDA: torch.cuda.synchronize()
    
    # ONNX
    onnx_path = f"baseline_gridsample_{case_id}.onnx"
    
    X_info = helper.make_tensor_value_info('X', onnx_dtype, [N, C, H_in, W_in])
    Grid_info = helper.make_tensor_value_info('Grid', onnx_dtype, [N, H_out, W_out, 2])
    Y_info = helper.make_tensor_value_info('Y', onnx_dtype, [N, C, H_out, W_out])
    
    node = helper.make_node(
        'GridSample',
        inputs=['X', 'Grid'],
        outputs=['Y'],
        align_corners=0,
        mode='linear',
        padding_mode='zeros'
    )
    
    graph = helper.make_graph([node], 'test', [X_info, Grid_info], [Y_info])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = 20 # GridSample was added in opset 16, updated in 20
    onnx.save(model, onnx_path)
    
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 3
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_CUDA else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, sess_opt, providers=providers)
    
    ort_inputs = {
        'X': x.detach().cpu().numpy(),
        'Grid': grid.detach().cpu().numpy()
    }
    
    ort_out = sess.run(['Y'], ort_inputs)[0]
    
    if os.path.exists(onnx_path): os.remove(onnx_path)
    
    # Compare
    torch_np = torch_out.detach().cpu().numpy()
    try:
        np.testing.assert_allclose(torch_np, ort_out, rtol=RTOL, atol=ATOL)
        return "SUCCESS"
    except AssertionError as e:
        diff = np.abs(torch_np - ort_out)
        return f"FAILURE (Max diff: {np.max(diff):.6f}, Mean: {np.mean(diff):.6f})"

def run_conv2d_test(case_id):
    # Conv2d Test
    N = random.randint(1, 4)
    C_in = random.randint(4, 16)
    C_out = random.randint(4, 16)
    H = random.randint(16, 32)
    W = random.randint(16, 32)
    K = 3
    
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if USE_FP16 else torch.float32
    onnx_dtype = TensorProto.FLOAT16 if USE_FP16 else TensorProto.FLOAT
    
    x = torch.randn(N, C_in, H, W, dtype=dtype, device=device) * 0.1
    weight = torch.randn(C_out, C_in, K, K, dtype=dtype, device=device) * 0.1
    bias = torch.randn(C_out, dtype=dtype, device=device) * 0.1
    
    # PyTorch
    if USE_CUDA: torch.cuda.synchronize()
    torch_out = F.conv2d(x, weight, bias, padding=1)
    if USE_CUDA: torch.cuda.synchronize()
    
    # ONNX
    onnx_path = f"baseline_conv_{case_id}.onnx"
    
    X_info = helper.make_tensor_value_info('X', onnx_dtype, [N, C_in, H, W])
    W_info = helper.make_tensor_value_info('W', onnx_dtype, [C_out, C_in, K, K])
    B_info = helper.make_tensor_value_info('B', onnx_dtype, [C_out])
    Y_info = helper.make_tensor_value_info('Y', onnx_dtype, [N, C_out, H, W])
    
    node = helper.make_node(
        'Conv',
        inputs=['X', 'W', 'B'],
        outputs=['Y'],
        kernel_shape=[K, K],
        pads=[1, 1, 1, 1]
    )
    
    graph = helper.make_graph([node], 'test', [X_info, W_info, B_info], [Y_info])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = 19
    onnx.save(model, onnx_path)
    
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 3
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_CUDA else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, sess_opt, providers=providers)
    
    ort_inputs = {
        'X': x.detach().cpu().numpy(),
        'W': weight.detach().cpu().numpy(),
        'B': bias.detach().cpu().numpy()
    }
    
    ort_out = sess.run(['Y'], ort_inputs)[0]
    
    if os.path.exists(onnx_path): os.remove(onnx_path)
    
    # Compare
    torch_np = torch_out.detach().cpu().numpy()
    try:
        np.testing.assert_allclose(torch_np, ort_out, rtol=RTOL, atol=ATOL)
        return "SUCCESS"
    except AssertionError as e:
        diff = np.abs(torch_np - ort_out)
        return f"FAILURE (Max diff: {np.max(diff):.6f}, Mean: {np.mean(diff):.6f})"

def main():
    print(f"Running Baseline Consistency Check")
    print(f"Mode: {'FP16' if USE_FP16 else 'FP32'}")
    print(f"CUDA: {USE_CUDA}")
    print(f"Tolerances: ATOL={ATOL}, RTOL={RTOL}")
    
    print("\n--- Testing GridSample (Similar to DeformConv interpolation) ---")
    failures = 0
    for i in range(NUM_TESTS):
        res = run_grid_sample_test(i)
        if res != "SUCCESS":
            print(f"GridSample Case {i}: {res}")
            failures += 1
    if failures == 0: print("GridSample: All passed.")
    
    print("\n--- Testing Conv2d (Standard Op) ---")
    failures = 0
    for i in range(NUM_TESTS):
        res = run_conv2d_test(i)
        if res != "SUCCESS":
            print(f"Conv2d Case {i}: {res}")
            failures += 1
    if failures == 0: print("Conv2d: All passed.")

if __name__ == "__main__":
    main()
