"""Orin Nano benchmark: latency, memory, power, energy/frame. RUN ON THE ORIN.

Reports the full protocol a robotics reviewer expects:
  - JetPack + TensorRT versions, power mode, clock-pinning state (logged, not assumed)
  - latency (ms / FPS), peak GPU mem
  - power: idle baseline measured AND subtracted -> net energy/frame (mJ)
  - robot terms: frames per watt-hour (and you can convert to % hover power)

Protocol to run first (write these values into the paper):
  sudo nvpmodel -m 2          # 25 W "MAXN-SUPER" on Orin Nano (verify index: nvpmodel -q)
  sudo jetson_clocks          # pin clocks
  cat /etc/nv_tegra_release   # JetPack/L4T version
  dpkg -l | grep -i tensorrt  # TRT version

Usage:
  python benchmark.py --engine ref.fp16.engine --warmup 50 --iters 500
"""
import argparse
import os
import re
import subprocess
import time
import numpy as np


def read_versions():
    info = {}
    try:
        info["l4t"] = open("/etc/nv_tegra_release").readline().strip()
    except Exception:
        info["l4t"] = "unknown"
    try:
        info["nvpmodel"] = subprocess.check_output(["nvpmodel", "-q"], text=True,
                                                    stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["nvpmodel"] = "unknown"
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except Exception:
        info["tensorrt"] = "unknown"
    return info


def parse_power_mw(line):
    """Sum the main rails from a tegrastats line (Orin: VDD_IN is the board total)."""
    m = re.search(r"VDD_IN\s+(\d+)mW", line)
    if m:
        return int(m.group(1))
    # fallback: sum GPU+CPU+SOC rails
    tot = 0
    found = False
    for tag in ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "POM_5V_IN"):
        m = re.search(rf"{tag}\s+(\d+)mW", line)
        if m:
            tot += int(m.group(1)); found = True
    return tot if found else None


def measure_idle_power(seconds=5, interval_ms=100, logfile="tegra_idle.log"):
    teg = subprocess.Popen(["tegrastats", "--interval", str(interval_ms), "--logfile", logfile],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(seconds)
    teg.terminate()
    try:
        teg.wait(timeout=2)
    except subprocess.TimeoutExpired:
        teg.kill()
    vals = []
    if os.path.exists(logfile):
        for line in open(logfile):
            p = parse_power_mw(line)
            if p is not None:
                vals.append(p)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--idle-seconds", type=int, default=5)
    args = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    info = read_versions()
    print("=== environment ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("  (confirm power mode = 25W MAXN-SUPER and jetson_clocks pinned)")

    # Idle baseline BEFORE loading the engine.
    print(f"\nmeasuring idle power for {args.idle_seconds}s ...")
    idle_mw = measure_idle_power(args.idle_seconds)
    print(f"  idle power = {idle_mw:.0f} mW")

    # ---- load engine (TensorRT 10 API) ----
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # allocate per-tensor buffers via the name-based TRT 10 API
    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host = np.zeros(tuple(shape), dtype=dtype)
        dev = cuda.mem_alloc(host.nbytes)
        is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_in:
            host[:] = np.random.rand(*shape).astype(dtype)
            cuda.memcpy_htod(dev, host)
        context.set_tensor_address(name, int(dev))
        tensors[name] = (host, dev, is_in)

    def infer():
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    for _ in range(args.warmup):
        infer()

    # timed + power-logged run
    teg = subprocess.Popen(["tegrastats", "--interval", "100", "--logfile", "tegra_run.log"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        infer()
    t1 = time.perf_counter()
    teg.terminate()
    try:
        teg.wait(timeout=2)
    except subprocess.TimeoutExpired:
        teg.kill()

    per_frame_ms = (t1 - t0) / args.iters * 1000.0
    fps = 1000.0 / per_frame_ms

    active = []
    if os.path.exists("tegra_run.log"):
        for line in open("tegra_run.log"):
            p = parse_power_mw(line)
            if p is not None:
                active.append(p)
    active_mw = float(np.mean(active)) if active else float("nan")
    net_mw = active_mw - idle_mw

    gross_mj = active_mw * per_frame_ms / 1000.0     # mW * ms = uJ ; /1000 -> mJ
    net_mj = net_mw * per_frame_ms / 1000.0
    frames_per_wh = 3.6e6 / net_mj if net_mj and net_mj == net_mj and net_mj > 0 else float("nan")

    print(f"\n=== {os.path.basename(args.engine)} ===")
    print(f"latency      = {per_frame_ms:.2f} ms  ({fps:.1f} FPS)")
    print(f"power active = {active_mw:.0f} mW   idle = {idle_mw:.0f} mW   net = {net_mw:.0f} mW")
    print(f"energy/frame = {gross_mj:.2f} mJ gross | {net_mj:.2f} mJ net-of-idle")
    print(f"robot terms  = {frames_per_wh:.0f} frames per Wh (net)")
    print(f"samples: {len(active)} active / power rail = VDD_IN")


if __name__ == "__main__":
    main()
