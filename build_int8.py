"""Build a TensorRT INT8 engine with entropy calibration. RUN ON THE ORIN.

INT8 is the single biggest energy knob on a Jetson, but cost-volume + soft-argmin
pipelines are rarely quantized — if it breaks disparity regression, that failure
is itself a finding. The --keep-fp16-output flag is the minimal remedy: keep the
soft-argmin / regression tail in higher precision while quantizing the rest.

Workflow:
  dev box:  python scripts/dump_calib.py --uwstereo-root ... --out calib_uw
            python export_tensorrt.py --config configs/ref.yaml --agg 3d --out onnx/ref.onnx --ckpt runs/ref-finetune/best.ckpt
            copy onnx/ + calib_uw/ to the Orin
  Orin:     python build_int8.py --onnx ref.onnx --calib calib_uw --out ref.int8.engine

Compare against the FP16 engine (trtexec --fp16) on accuracy + energy.
"""
import argparse
import os
import glob
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True, help="folder of *_left.npy / *_right.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="int8_calib.cache")
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--keep-fp16-output", action="store_true",
                    help="if INT8 wrecks disparity, keep the output layers in FP16")
    args = ap.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 fallback for unsupported INT8 ops

    # ---- entropy calibrator reading the dumped .npy pairs ----
    lefts = sorted(glob.glob(os.path.join(args.calib, "*_left.npy")))
    rights = sorted(glob.glob(os.path.join(args.calib, "*_right.npy")))
    assert lefts and len(lefts) == len(rights), "calib set empty or mismatched"

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self.cuda = cuda
            self.idx = 0
            self.cache = args.cache
            l0 = np.load(lefts[0]); r0 = np.load(rights[0])
            self.dl = cuda.mem_alloc(l0.nbytes)
            self.dr = cuda.mem_alloc(r0.nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(lefts):
                return None
            l = np.ascontiguousarray(np.load(lefts[self.idx]))
            r = np.ascontiguousarray(np.load(rights[self.idx]))
            self.cuda.memcpy_htod(self.dl, l)
            self.cuda.memcpy_htod(self.dr, r)
            self.idx += 1
            return [int(self.dl), int(self.dr)]

        def read_calibration_cache(self):
            return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

        def write_calibration_cache(self, cache):
            with open(self.cache, "wb") as f:
                f.write(cache)

    config.int8_calibrator = Calib()

    # ---- optional remedy: pin the output (disparity regression) layers to FP16 ----
    if args.keep_fp16_output:
        last = network.get_layer(network.num_layers - 1)
        for li in range(max(0, network.num_layers - 6), network.num_layers):
            layer = network.get_layer(li)
            layer.precision = trt.float16
            for oi in range(layer.num_outputs):
                layer.set_output_type(oi, trt.float16)
        print("pinned last ~6 layers to FP16 (regression-head remedy)")

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("INT8 engine build failed")
    with open(args.out, "wb") as f:
        f.write(engine)
    print(f"wrote INT8 engine: {args.out}")


if __name__ == "__main__":
    main()
