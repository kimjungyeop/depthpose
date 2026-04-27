"""Frame-quality filters for oracle parquets.

Currently: side-consistency check for the L/R skeleton labels. ViTPose
sometimes swaps the left/right assignment at a single joint, producing
an X-shaped skeleton across the hip→knee or knee→ankle edges. Such
frames are unreliable training targets; we exclude them from splits.

Detection is geometric: for each pair (hip, knee, ankle) we look at
``sign(left.u_px - right.u_px)``. In a non-swapped skeleton the same
side of the body keeps the same image-x ordering across all three
pairs. If the sign flips between any two pairs, the labels crossed.
"""

from __future__ import annotations

import pandas as pd

_PAIR_NAMES: tuple[str, ...] = ("hip", "knee", "ankle")


def is_frame_skeleton_consistent(frame_rows: pd.DataFrame) -> bool:
    """True if the L/R label assignment is consistent across hip/knee/ankle.

    ``frame_rows`` should hold the joint rows for a single frame, with
    at least ``joint_name`` and ``u_px`` columns.
    """
    by_joint = frame_rows.set_index("joint_name")
    signs: list[int] = []
    for p in _PAIR_NAMES:
        ln, rn = f"left_{p}", f"right_{p}"
        if ln in by_joint.index and rn in by_joint.index:
            d = float(by_joint.at[ln, "u_px"]) - float(by_joint.at[rn, "u_px"])
            if d != 0.0:
                signs.append(1 if d > 0 else -1)
    if len(signs) < 2:
        # not enough joints to detect a swap; treat as consistent
        return True
    return len(set(signs)) == 1


def consistent_frame_indices(parquet_df: pd.DataFrame) -> dict[int, bool]:
    """``frame_index`` → True/False under the side-consistency rule."""
    out: dict[int, bool] = {}
    for fi, group in parquet_df.groupby("frame_index"):
        out[int(fi)] = is_frame_skeleton_consistent(group)
    return out
