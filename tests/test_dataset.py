"""Tests for WalkerSession Dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from depthpose.data.dataset import WalkerSession


_RAW = Path("data/raw")
_SPLIT = Path("data/splits/random_train0.8_seed0.json")


def _first_meta_path() -> Path | None:
    metas = sorted(_RAW.glob("*/*/meta.json"))
    return metas[0] if metas else None


def _has_pilot() -> bool:
    return _first_meta_path() is not None


@pytest.mark.skipif(not _has_pilot(), reason="no extracted sessions")
def test_dataset_len_matches_meta() -> None:
    ds = WalkerSession(_RAW)
    total = 0
    for mp in sorted(_RAW.glob("*/*/meta.json")):
        total += json.loads(mp.read_text())["saved_frames"]
    assert len(ds) == total


@pytest.mark.skipif(not _has_pilot(), reason="no extracted sessions")
def test_dataset_returns_correct_dtypes_and_shapes() -> None:
    ds = WalkerSession(_RAW)
    sample = ds[0]
    meta_path = _first_meta_path()
    assert meta_path is not None
    meta = json.loads(meta_path.read_text())
    H = meta["depth_intrinsics"]["height"]
    W = meta["depth_intrinsics"]["width"]
    assert sample["rgb"].dtype == np.uint8
    assert sample["rgb"].shape == (H, W, 3)
    assert sample["depth_mm"].dtype == np.uint16
    assert sample["depth_mm"].shape == (H, W)
    assert isinstance(sample["intrinsics"], tuple)
    assert len(sample["intrinsics"]) == 4
    fx, fy, cx, cy = sample["intrinsics"]
    assert fx == meta["depth_intrinsics"]["fx"]
    assert fy == meta["depth_intrinsics"]["fy"]
    assert cx == meta["depth_intrinsics"]["cx"]
    assert cy == meta["depth_intrinsics"]["cy"]
    assert sample["color_intrinsics"] == sample["depth_intrinsics"]
    assert sample["subject"] == str(meta["subject"])
    assert sample["session"] == str(meta["session"])
    assert sample["frame_index"] == 0


@pytest.mark.skipif(not _has_pilot(), reason="pilot session not extracted")
def test_dataset_deterministic_load() -> None:
    ds = WalkerSession(_RAW)
    a = ds[0]
    b = ds[0]
    assert np.array_equal(a["rgb"], b["rgb"])
    assert np.array_equal(a["depth_mm"], b["depth_mm"])


@pytest.mark.skipif(not _has_pilot(), reason="pilot session not extracted")
def test_dataset_deterministic_order() -> None:
    ds = WalkerSession(_RAW)
    n = min(20, len(ds))
    items = [(ds[i]["subject"], ds[i]["session"], ds[i]["frame_index"]) for i in range(n)]
    assert items == sorted(items)


@pytest.mark.skipif(not (_has_pilot() and _SPLIT.exists()), reason="pilot/splits missing")
def test_dataset_split_train_test_disjoint_and_complete() -> None:
    ds_train = WalkerSession(_RAW, split_file=_SPLIT, split="train")
    ds_test = WalkerSession(_RAW, split_file=_SPLIT, split="test")
    ds_all = WalkerSession(_RAW, split_file=_SPLIT, split="all")
    assert len(ds_train) + len(ds_test) == len(ds_all)
    train_ids = {(ds_train[i]["subject"], ds_train[i]["session"], ds_train[i]["frame_index"])
                 for i in range(len(ds_train))}
    test_ids = {(ds_test[i]["subject"], ds_test[i]["session"], ds_test[i]["frame_index"])
                for i in range(len(ds_test))}
    assert train_ids.isdisjoint(test_ids)
    all_ids = {(ds_all[i]["subject"], ds_all[i]["session"], ds_all[i]["frame_index"])
               for i in range(len(ds_all))}
    assert train_ids | test_ids == all_ids


@pytest.mark.skipif(not (_has_pilot() and _SPLIT.exists()), reason="pilot/splits missing")
def test_dataset_split_ratio_about_train_frac() -> None:
    payload = json.loads(_SPLIT.read_text())
    train_frac = payload["train_frac"]
    ds_train = WalkerSession(_RAW, split_file=_SPLIT, split="train")
    ds_test = WalkerSession(_RAW, split_file=_SPLIT, split="test")
    n_total = len(ds_train) + len(ds_test)
    assert abs(len(ds_train) / n_total - train_frac) < 0.01


def test_dataset_unknown_session_raises(tmp_path: Path) -> None:
    (tmp_path / "S99" / "9").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        WalkerSession(tmp_path, sessions=[("S99", "9")])


def test_dataset_empty_raw_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        WalkerSession(tmp_path)
