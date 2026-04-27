"""Export the trained student to ONNX + verify parity with the PyTorch model.

Outputs into the run directory:
- ``student.onnx``           the exported graph (opset 17, dynamic batch).
- ``student_onnx_meta.json`` size, parameter count, op set, parity diagnostics.

The student takes two inputs (``depth``, ``intrinsics_input``) and returns
``coords_3d`` (B, J, 3) — the only output we need at inference time. The
auxiliary heatmaps are not exported.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import typer

from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


class _ExportWrapper(torch.nn.Module):
    """Wraps the student so ONNX export sees a single Tensor output."""

    def __init__(self, student: DepthPoseStudent) -> None:
        super().__init__()
        self.student = student

    def forward(
        self,
        depth: torch.Tensor,
        intrinsics_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.student(depth, intrinsics_input)["coords_3d"]


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint", help="best | last"),
    opset: int = typer.Option(17, "--opset"),
    dynamic_batch: bool = typer.Option(True, "--dynamic-batch/--static-batch"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.data is not None

    # CPU export so the ONNX graph isn't tagged with CUDA-only ops.
    device = "cpu"
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone,
        num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv,
        deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta,
        pretrained=False,
    ).to(device)
    state = torch.load(run_dir / f"{checkpoint}.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = _ExportWrapper(model).eval()

    W, H = cfg.data.image_size  # configs are stored (W, H)
    depth = torch.randn(1, 1, H, W)
    # Plausible intrinsics for the resized depth frame.
    intr = torch.tensor([[200.0, 200.0, W / 2, H / 2]])

    onnx_path = run_dir / "student.onnx"
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "depth": {0: "batch"},
            "intrinsics_input": {0: "batch"},
            "coords_3d": {0: "batch"},
        }

    logger.info("exporting → %s (opset %d, dynamic_batch=%s)",
                onnx_path, opset, dynamic_batch)
    torch.onnx.export(
        wrapper,
        (depth, intr),
        str(onnx_path),
        opset_version=opset,
        input_names=["depth", "intrinsics_input"],
        output_names=["coords_3d"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter; new dynamo path needs onnxscript
    )

    # Validate ONNX graph + report op count / size.
    import onnx
    g = onnx.load(str(onnx_path))
    onnx.checker.check_model(g)
    n_nodes = len(g.graph.node)
    op_types = sorted({n.op_type for n in g.graph.node})
    onnx_bytes = onnx_path.stat().st_size
    n_params = sum(p.numel() for p in model.parameters())

    # Parity check via onnxruntime.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    diffs: list[float] = []
    for _ in range(8):
        d_np = rng.standard_normal((1, 1, H, W)).astype(np.float32)
        i_np = np.array([[200.0, 200.0, W / 2, H / 2]], dtype=np.float32)
        with torch.inference_mode():
            torch_out = wrapper(torch.from_numpy(d_np), torch.from_numpy(i_np)).numpy()
        ort_out, = sess.run(["coords_3d"], {"depth": d_np, "intrinsics_input": i_np})
        diffs.append(float(np.max(np.abs(torch_out - ort_out))))
    max_abs_diff = max(diffs)
    median_abs_diff = float(np.median(diffs))

    meta = {
        "onnx_path": str(onnx_path),
        "checkpoint": str(run_dir / f"{checkpoint}.pt"),
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "input_shape_depth": [1, 1, H, W],
        "image_size_wh": [W, H],
        "n_params_pytorch": int(n_params),
        "onnx_size_mb": round(onnx_bytes / 1024 / 1024, 3),
        "n_onnx_nodes": n_nodes,
        "onnx_op_types": op_types,
        "parity_max_abs_diff_mm": max_abs_diff * 1000.0,
        "parity_median_abs_diff_mm": median_abs_diff * 1000.0,
    }
    (run_dir / "student_onnx_meta.json").write_text(json.dumps(meta, indent=2))

    typer.echo("\n=== ONNX export ===")
    typer.echo(f"file:           {onnx_path}")
    typer.echo(f"size:           {meta['onnx_size_mb']} MB")
    typer.echo(f"params (torch): {n_params:,}")
    typer.echo(f"nodes:          {n_nodes}, op_types: {len(op_types)}")
    typer.echo(f"parity max |Δ|: {meta['parity_max_abs_diff_mm']:.3f} mm "
               f"(median {meta['parity_median_abs_diff_mm']:.3f} mm)")


if __name__ == "__main__":
    app()
