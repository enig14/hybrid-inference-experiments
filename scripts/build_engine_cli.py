import argparse
from pathlib import Path

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import os


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class NpyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, npy_path, cache_file="calib.cache", *, mmap=True):
        super().__init__()
        self.npy_path = npy_path
        self.batch_size = 1  # [고정] 무조건 1장씩 처리
        self.cache_file = cache_file
        self.idx = 0

        # 대용량 데이터도 mmap으로 안전하게 로드
        self.data = np.load(npy_path, mmap_mode="r" if mmap else None)

        # --------------------------
        # Public-repo friendly checks
        # --------------------------
        # Expect NCHW, RGB: (N,3,H,W)
        if getattr(self.data, "ndim", None) != 4:
            raise ValueError(
                f"[ERR] calib_npy must be 4D (N,3,H,W). "
                f"Got shape={getattr(self.data, 'shape', None)} from {npy_path}"
            )
        if self.data.shape[1] != 3:
            raise ValueError(
                f"[ERR] calib_npy must be NCHW with C=3 (RGB). "
                f"Got shape={self.data.shape} from {npy_path}"
            )

        # [최적화] GPU 메모리를 __init__에서 딱 한 번만 할당합니다.
        # 데이터 1장의 크기 (Byte) = C * H * W * 4 (float32 size)
        one_frame_shape = self.data.shape[1:]  # (3, H, W)
        self.nbytes = int(np.prod(one_frame_shape) * 4)

        # GPU 메모리 할당 (malloc)
        self.d_input = cuda.mem_alloc(self.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # 데이터 끝에 도달하면 None 반환 (종료 신호)
        if self.idx >= self.data.shape[0]:
            return None

        # [최적화] 슬라이싱 없이 인덱스로 바로 접근 (속도 향상)
        # (3, H, W) 한 장을 가져와서 float32로 변환
        batch = self.data[self.idx].astype(np.float32)
        self.idx += 1

        # 메모리 연속성 보장 (1차원 배열로 펼침)
        batch = np.ascontiguousarray(batch.ravel())

        # Host -> Device 복사 (미리 할당된 주소로 덮어쓰기)
        cuda.memcpy_htod(self.d_input, batch)

        # 포인터 반환
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # 캐시 파일이 있으면 읽어서 반환 (재사용)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        # 캘리브레이션 결과를 캐시 파일로 저장
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def _find_tensor_by_name(network: trt.INetworkDefinition, name: str):
    """Find a tensor by exact name across inputs, existing outputs, and layer outputs."""
    # inputs
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if t is not None and t.name == name:
            return t

    # existing outputs
    for i in range(network.num_outputs):
        t = network.get_output(i)
        if t is not None and t.name == name:
            return t

    # layer outputs
    for li in range(network.num_layers):
        layer = network.get_layer(li)
        for oi in range(layer.num_outputs):
            t = layer.get_output(oi)
            if t is not None and t.name == name:
                return t

    return None


def _mark_debug_outputs(network: trt.INetworkDefinition, names_csv: str):
    """
    Mark specified tensors as network outputs.
    - Comma-separated list of exact tensor names.
    - If a name cannot be resolved, raise (avoid silent no-op).
    """
    names = [s.strip() for s in (names_csv or "").split(",") if s.strip()]
    if not names:
        return

    existing = set()
    for i in range(network.num_outputs):
        t = network.get_output(i)
        if t is not None:
            existing.add(t.name)

    for n in names:
        if n in existing:
            continue
        t = _find_tensor_by_name(network, n)
        if t is None:
            raise ValueError(f"[ERR] debug_outputs tensor not found in network: '{n}'")
        network.mark_output(t)
        existing.add(n)


def build_engine_trt10(
    onnx_path: str,
    calib_npy: str,
    engine_path: str,
    cache_path: str = "calib.cache",
    dla_core: int = 0,
    workspace_bytes: int = (2 << 30),
    calib_mmap: bool = True,
    precision: str = "int8",
    no_dla: bool = False,
    debug_outputs: str = "",
):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    # 실제로 debug_outputs가 동작하도록: 지정 텐서를 output으로 마킹
    _mark_debug_outputs(network, debug_outputs)

    config = builder.create_builder_config()

    # Workspace (TRT 10)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    except Exception:
        # Older fallback
        config.max_workspace_size = int(workspace_bytes)

    # INT8 + calibrator
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = NpyCalibrator(calib_npy, cache_file=cache_path, mmap=calib_mmap)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    # DLA + GPU fallback
    if not no_dla:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla_core)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # TRT 10: build serialized engine bytes
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed (serialized is None)")

    # Save engine
    engine_path = str(engine_path)
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        # IHostMemory is often bytes-like, but handle both cases.
        try:
            f.write(serialized)
        except TypeError:
            f.write(bytearray(serialized))

    print("[OK] saved engine:", engine_path)
    if precision == "int8":
        print("[OK] calib cache path:", cache_path)
    if debug_outputs:
        print("[OK] debug_outputs:", debug_outputs)

def _parse_workspace(v: str) -> int:
    """
    Accept either raw integer bytes, or suffixed forms like: 2147483648, 2GiB, 2048MiB, 2GB, 2000MB.
    Uses binary for KiB/MiB/GiB, decimal for KB/MB/GB.
    """
    s = v.strip()
    # pure int bytes
    if s.isdigit():
        return int(s)

    units = {
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
    }
    s_low = s.lower()
    for u, mul in units.items():
        if s_low.endswith(u):
            num = s_low[: -len(u)].strip()
            try:
                return int(float(num) * mul)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"invalid workspace '{v}'") from e
    raise argparse.ArgumentTypeError(f"invalid workspace '{v}' (use bytes or KiB/MiB/GiB/KB/MB/GB)")


def main():
    ap = argparse.ArgumentParser(description="Build TensorRT 10 serialized engine (INT8, DLA+GPU fallback) from ONNX.")
    ap.add_argument("--onnx_path", required=True, help="Input ONNX path")
    ap.add_argument("--calib_npy", default="", help="Calibration NPY path (N,3,H,W). Required for --precision int8.")
    ap.add_argument("--cache_path", default="", help="Calibration cache output/read path. Required for --precision int8.")
    ap.add_argument("--engine_path", required=True, help="Output engine/plan path")

    ap.add_argument("--dla_core", type=int, default=0, help="DLA core index (default: 0)")
    ap.add_argument("--workspace", type=_parse_workspace, default=(2 << 30), help="Workspace size (e.g., 2GiB)")

    ap.add_argument("--no_calib_mmap", action="store_true", help="Disable np.load mmap (loads full .npy into RAM)")
    ap.add_argument("--debug_outputs", type=str, default="", help="comma-separated tensor names to mark as network outputs")
    ap.add_argument("--precision", choices=["int8", "fp16"], default="int8",
                    help="Build precision mode (int8 uses calibrator, fp16 builds float probe)")
    ap.add_argument("--no_dla", action="store_true",
                    help="Disable DLA for probe (force GPU). Recommended for fp16 probe.")
    args = ap.parse_args()

    # basic validation
    if not Path(args.onnx_path).is_file():
        raise SystemExit(f"[ERR] onnx_path not found: {args.onnx_path}")

    if args.precision == "int8":
        if not args.calib_npy:
            raise SystemExit("[ERR] --calib_npy is required when --precision int8")
        if not args.cache_path:
            raise SystemExit("[ERR] --cache_path is required when --precision int8")
        if not Path(args.calib_npy).is_file():
            raise SystemExit(f"[ERR] calib_npy not found: {args.calib_npy}")
    else:
        # fp16: ignore calib/cache if not provided (allowed)
        if args.calib_npy and (not Path(args.calib_npy).is_file()):
            raise SystemExit(f"[ERR] calib_npy not found: {args.calib_npy}")

    build_engine_trt10(
        onnx_path=args.onnx_path,
        calib_npy=args.calib_npy,
        engine_path=args.engine_path,
        cache_path=args.cache_path,
        dla_core=args.dla_core,
        workspace_bytes=args.workspace,
        calib_mmap=(not args.no_calib_mmap),
        precision=args.precision,
        no_dla=args.no_dla,
        debug_outputs=args.debug_outputs,
    )


if __name__ == "__main__":
    main()