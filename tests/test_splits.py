"""Regression tests for make_splits: determinism, coverage, isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from depthpose.data.make_splits import build_splits


def _make_fake_session(root: Path, subject: str, session: str, n_frames: int) -> None:
    sd = root / subject / session
    sd.mkdir(parents=True)
    (sd / "rgb").mkdir()
    (sd / "depth").mkdir()
    (sd / "meta.json").write_text(json.dumps({
        "subject": subject,
        "session": session,
        "saved_frames": n_frames,
    }))


def test_disjoint_full_coverage(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "S01", "1", 200)
    s = build_splits(tmp_path, train_frac=0.8, seed=0)["splits"]["S01/1"]
    train, test = set(s["train"]), set(s["test"])
    assert train.isdisjoint(test)
    assert train | test == set(range(200))


def test_ratio_at_round_sizes(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "S01", "1", 1000)
    s = build_splits(tmp_path, train_frac=0.8, seed=0)["splits"]["S01/1"]
    assert len(s["train"]) == 800
    assert len(s["test"]) == 200


def test_deterministic_same_seed(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "S01", "1", 500)
    a = build_splits(tmp_path, train_frac=0.8, seed=0)
    b = build_splits(tmp_path, train_frac=0.8, seed=0)
    assert a == b


def test_seed_changes_assignment(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "S01", "1", 500)
    a = build_splits(tmp_path, train_frac=0.8, seed=0)["splits"]["S01/1"]
    b = build_splits(tmp_path, train_frac=0.8, seed=1)["splits"]["S01/1"]
    assert a != b


def test_per_session_isolation(tmp_path: Path) -> None:
    """Adding a new session must not change an existing session's split."""
    _make_fake_session(tmp_path, "S01", "1", 200)
    before = build_splits(tmp_path, train_frac=0.8, seed=0)
    _make_fake_session(tmp_path, "S02", "1", 100)
    after = build_splits(tmp_path, train_frac=0.8, seed=0)
    assert before["splits"]["S01/1"] == after["splits"]["S01/1"]


def test_rejects_bad_train_frac(tmp_path: Path) -> None:
    _make_fake_session(tmp_path, "S01", "1", 200)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            build_splits(tmp_path, train_frac=bad, seed=0)


def test_no_sessions_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_splits(tmp_path, train_frac=0.8, seed=0)
