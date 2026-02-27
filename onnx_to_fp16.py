"""
将 ONNX 模型转换为 FP16，减小体积并利于 GPU 推理。
用法: python onnx_to_fp16.py [输入.onnx] [输出.onnx]
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ONNX 模型转 FP16")
    parser.add_argument(
        "input",
        nargs="?",
        default="BiRefNet_HR-matting-epoch_135.onnx",
        help="输入 ONNX 文件路径（默认: BiRefNet_HR-matting-epoch_135.onnx）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 FP16 ONNX 文件路径（默认: 输入名_fp16.onnx）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # 尝试在 models 目录下查找
        alt = Path("models") / input_path.name
        if alt.exists():
            input_path = alt
        else:
            print(f"错误: 找不到文件 {args.input}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_stem(
        input_path.stem + "_fp16"
    )
    output_path = output_path.with_suffix(".onnx")

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("请先安装: pip install onnx onnxconverter-common", file=sys.stderr)
        sys.exit(1)

    print("正在加载 ONNX 模型...")
    model = onnx.load(str('BiRefNet_HR-matting-epoch_135.slim.onnx'))
    print("正在转换为 FP16...")
    # disable_shape_infer=True：大模型不做整图 shape 推断，避免卡死/假死
    # keep_io_types: 输入/输出保持 float32
    # op_block_list: 下列算子保持 float32，避免 DML 下类型不匹配；Cast/Softmax 必须保留，其余是库默认
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    print("正在保存...")
    onnx.save(model_fp16, str(output_path))
    print(f"已保存 FP16 模型: {output_path}")


if __name__ == "__main__":
    main()
