"""Latency benchmark for the depth-only student.

Measures inference latency under three runtimes:
- PyTorch CPU
- PyTorch CUDA (skipped if no GPU)
- ONNX Runtime CPU (with the exported ``student.onnx``)

Reports per-frame latency (mean / median / p95) over a configurable number
of warmup + measured iterations at batch size 1 (the deployment scenario).
Writes a JSON sidecar next to the ONNX file.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
import typer

from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _bench(fn, n_warmup: int, n_iter: int, sync=lambda: None) -> dict[str, float]:
    for _ in range(n_warmup):
        fn(); sync()
    samples_ms: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        sync()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    return {
        "n_iter": n_iter,
        "mean_ms": float(mean(samples_ms)),
        "median_ms": float(median(samples_ms)),
        "p95_ms": float(samples_ms[max(0, int(n_iter * 0.95) - 1)]),
        "min_ms": float(samples_ms[0]),
        "max_ms": float(samples_ms[-1]),
    }


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint"),
    n_warmup: int = typer.Option(10, "--n-warmup"),
    n_iter: int = typer.Option(50, "--n-iter"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.data is not None
    W, H = cfg.data.image_size

    # Build the model once.
    def _build_pytorch(device: str) -> DepthPoseStudent:
        m = DepthPoseStudent(
            backbone_name=cfg.model.backbone,
            num_joints=cfg.model.num_joints,
            num_deconv=cfg.model.num_deconv,
            deconv_channels=cfg.model.deconv_channels,
            softargmax_beta=cfg.model.softargmax_beta,
            pretrained=False,
        ).to(device)
        state = torch.load(run_dir / f"{checkpoint}.pt",
                           map_location=device, weights_only=True)
        m.load_state_dict(state["model"])
        m.eval()
        return m

    rng = np.random.default_rng(0)
    depth_np = rng.standard_normal((1, 1, H, W)).astype(np.float32)
    intr_np = np.array([[200.0, 200.0, W / 2, H / 2]], dtype=np.float32)

    results: dict[str, dict] = {}

    # --- PyTorch CPU ---
    logger.info("benchmarking PyTorch CPU…")
    m_cpu = _build_pytorch("cpu")
    d_cpu = torch.from_numpy(depth_np); i_cpu = torch.from_numpy(intr_np)
    @torch.inference_mode()
    def _torch_cpu():
        m_cpu(d_cpu, i_cpu)
    results["pytorch_cpu"] = _bench(_torch_cpu, n_warmup, n_iter)

    # --- PyTorch CUDA ---
    if torch.cuda.is_available():
        logger.info("benchmarking PyTorch CUDA…")
        m_cuda = _build_pytorch("cuda")
        d_g = torch.from_numpy(depth_np).cuda(); i_g = torch.from_numpy(intr_np).cuda()
        @torch.inference_mode()
        def _torch_cuda():
            m_cuda(d_g, i_g)
        results["pytorch_cuda"] = _bench(
            _torch_cuda, n_warmup, n_iter, sync=torch.cuda.synchronize,
        )
    else:
        logger.info("CUDA not available, skipping pytorch_cuda")

    # --- ONNX Runtime CPU ---
    onnx_path = run_dir / "student.onnx"
    if onnx_path.exists():
        logger.info("benchmarking ONNX Runtime CPU…")
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1  # single-threaded — closer to embedded
        sess = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        feed = {"depth": depth_np, "intrinsics_input": intr_np}
        def _ort():
            sess.run(["coords_3d"], feed)
        results["onnxruntime_cpu_1thread"] = _bench(_ort, n_warmup, n_iter)

        # Also benchmark with all available threads.
        sess_opts_mt = ort.SessionOptions()
        sess_mt = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts_mt,
            providers=["CPUExecutionProvider"],
        )
        def _ort_mt():
            sess_mt.run(["coords_3d"], feed)
        results["onnxruntime_cpu_default_threads"] = _bench(_ort_mt, n_warmup, n_iter)
    else:
        logger.warning("ONNX file %s not found; skipping ORT bench", onnx_path)

    # Add device / system info.
    import platform
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": int(__import__("os").cpu_count() or -1),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_size_wh": [W, H],
    }
    payload = {"system": info, "results": results}

    out_path = run_dir / "latency_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2))

    typer.echo("\n=== latency benchmark (batch size 1) ===")
    typer.echo(f"input: depth ({1},{1},{H},{W}) | iters: {n_iter} (warmup {n_warmup})")
    typer.echo(f"{'runtime':<32} {'mean':>8} {'median':>8} {'p95':>8} {'fps@median':>10}")
    for k, v in results.items():
        fps = 1000.0 / max(v['median_ms'], 1e-6)
        typer.echo(f"{k:<32} {v['mean_ms']:>7.2f}m {v['median_ms']:>7.2f}m "
                   f"{v['p95_ms']:>7.2f}m {fps:>9.1f}")
    typer.echo(f"\nwrote {out_path}")


if __name__ == "__main__":
    app()
