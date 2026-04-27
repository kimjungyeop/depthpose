"""Pydantic schema for the single-YAML-per-run config.

Phase 1 only populates `project`, `data`, and `oracle`. Later phases
will fill in `model`, `training`, and `eval`. Each section is optional
at the top level so a Phase-1 config can omit the unbuilt parts without
failing validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Strict):
    name: str
    seed: int = 0


class DataConfig(_Strict):
    raw_dir: Path
    labels_dir: Path
    keep_every: Annotated[int, Field(ge=1)] = 1  # 1 = no subsampling (default)
    image_size: tuple[int, int] = (192, 256)  # (W, H), portrait — matches walker depth aspect
    splits_path: Path | None = None

    @model_validator(mode="after")
    def _coerce_paths(self) -> DataConfig:  # noqa: D401
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        object.__setattr__(self, "labels_dir", Path(self.labels_dir))
        if self.splits_path is not None:
            object.__setattr__(self, "splits_path", Path(self.splits_path))
        return self


class ModelConfig(_Strict):
    backbone: Literal["mobilenetv2_100"] = "mobilenetv2_100"
    num_joints: Annotated[int, Field(ge=1)] = 6
    num_deconv: Annotated[int, Field(ge=1, le=4)] = 3
    deconv_channels: Annotated[int, Field(ge=16)] = 256
    softargmax_beta: float = 100.0  # sharpness; higher = closer to argmax


class TrainingConfig(_Strict):
    batch_size: Annotated[int, Field(ge=1)] = 64
    num_epochs: Annotated[int, Field(ge=1)] = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_pct: Annotated[float, Field(ge=0.0, le=0.5)] = 0.05
    num_workers: Annotated[int, Field(ge=0)] = 4
    log_dir: Path = Path("runs")
    aux_heatmap_loss_weight: float = 0.0  # 0 = pure 3D loss; >0 enables aux 2D MSE
    anatomical_loss_weight: float = 0.0   # 0 = off; >0 penalises L/R lateral crossover
    grad_clip_norm: float = 1.0


class EvalConfig(_Strict):
    pck_thresholds_mm: tuple[int, ...] = (5, 10, 20, 50)


JointName = Literal[
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_tip",
    "right_foot_tip",
]


class OracleConfig(_Strict):
    backend: Literal["mmpose", "huggingface"] = "mmpose"
    checkpoint: str = "vitpose-plus-base-cocowholebody"
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    median_kernel: Annotated[int, Field(ge=1)] = 3
    joints: tuple[JointName, ...] = (
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_tip", "right_foot_tip",
    )


class Config(_Strict):
    project: ProjectConfig
    data: DataConfig
    oracle: OracleConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    eval: EvalConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
