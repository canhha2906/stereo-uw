"""Stage 6b: build TensorRT engines (FP16 / INT8) from ONNX, on the dev box.

Paper 1's build_int8.py needs pycuda, which wants a working MSVC toolchain on Windows.
Torch + CUDA already work here, so the calibrator below uses torch tensors for device
memory and pycuda is not required.

INT8 calibration uses RENDERED KITTI, never UWStereo. Calibration fits quantisation
scales to data, so calibrating on the underwater set would leak the target domain and
break the non-negotiable in CLAUDE.md that underwater data is evaluation only.

Note this builds for the dev GPU (RTX 5060). Engines are NOT portable to the Orin -
the Orin must build its own from the same ONNX. What transfers is the accuracy finding:
whether INT8 preserves disparity regression is a property of the network, not the chip.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


class TorchCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, cache, limit=0):
        super().__init__()
        self.cache_path = Path(cache)
        files = sorted(Path(calib_dir).glob("*_left.npy"))
        if limit:
            files = files[:limit]
        self.files = files
        self.i = 0
        s = np.load(files[0])
        self.shape = s.shape
        self.dL = torch.empty(int(np.prod(s.shape)), dtype=torch.float32, device="cuda")
        self.dR = torch.empty(int(np.prod(s.shape)), dtype=torch.float32, device="cuda")
        print(f"calibrator: {len(files)} pairs, input {s.shape}")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= len(self.files):
            return None
        lp = self.files[self.i]
        rp = lp.with_name(lp.name.replace("_left", "_right"))
        self.dL.copy_(torch.from_numpy(np.load(lp).ravel()).cuda())
        self.dR.copy_(torch.from_numpy(np.load(rp).ravel()).cuda())
        self.i += 1
        return [int(self.dL.data_ptr()), int(self.dR.data_ptr())]

    def read_calibration_cache(self):
        return self.cache_path.read_bytes() if self.cache_path.exists() else None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


def build(onnx_path, out_path, precision, calib_dir=None, cache=None, workspace_mb=2048,
          calib_limit=0):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()          # TRT 10: always explicit batch
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        for i in range(parser.num_errors):
            print("  parser:", parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    calib = None
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.FP16)       # FP16 fallback for layers INT8 can't take
        calib = TorchCalibrator(calib_dir, cache, calib_limit)
        cfg.int8_calibrator = calib

    print(f"building {precision} engine from {onnx_path} ...", flush=True)
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise SystemExit(f"{precision} engine build FAILED")
    Path(out_path).write_bytes(plan)
    print(f"wrote {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    ap.add_argument("--calib", default="calib_rendered_kitti")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--calib-limit", type=int, default=0)
    a = ap.parse_args()
    build(a.onnx, a.out, a.precision, a.calib,
          a.cache or f"{Path(a.out).stem}.cache",
          a.workspace_mb, a.calib_limit)
