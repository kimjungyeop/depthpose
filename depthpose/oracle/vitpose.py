"""HuggingFace ViTPose++ wrapper for the RGB oracle.

Loads a ViTPose+ multi-task checkpoint (default: ``usyd-community/
vitpose-plus-base``) and exposes a thin ``ViTPoseOracle.detect`` API
that returns the 10 lower-body keypoints we care about, in COCO-
WholeBody coordinates within the image frame supplied.

The model takes a *cropped person bbox* per image. For our walker view
the person fills the frame, so the default bbox is ``(0, 0, W, H)``.
A future person-detector swap is one line in the constructor.
"""

from __future__ import annotations

import dataclasses
import logging

import cv2
import numpy as np
import torch
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

logger = logging.getLogger(__name__)


# Multi-task ViTPose+ dataset indices (per HF model card).
DATASET_COCO = 0
DATASET_AIC = 1
DATASET_MPII = 2
DATASET_COCO_WHOLEBODY = 3


# COCO-17 indices for the 6 lower-body joints the standard ViTPose head
# returns. The 10-joint set (incl. heel + foot-tip) requires a COCO-
# WholeBody multi-task checkpoint, which `usyd-community/vitpose-plus-base`
# is not — its head is single COCO-17.
JOINTS_COCO17: dict[str, int] = {
    "left_hip":    11,
    "right_hip":   12,
    "left_knee":   13,
    "right_knee":  14,
    "left_ankle":  15,
    "right_ankle": 16,
}

# COCO-WholeBody mapping kept as a forward-looking constant for when we
# swap in a wholebody-trained checkpoint.
JOINTS_COCO_WHOLEBODY: dict[str, int] = {
    "left_hip":       11,
    "right_hip":      12,
    "left_knee":      13,
    "right_knee":     14,
    "left_ankle":     15,
    "right_ankle":    16,
    "left_heel":      19,
    "right_heel":     22,
    "left_foot_tip":  17,
    "right_foot_tip": 20,
}


@dataclasses.dataclass
class Keypoints2D:
    """Per-frame 2D keypoint detections + confidences."""
    joint_names: list[str]
    coords: np.ndarray  # (J, 2) float32, (u, v) in image pixels
    scores: np.ndarray  # (J,) float32, ViTPose confidence in [0, 1]


class ViTPoseOracle:
    """ViTPose++ → 2D keypoints (COCO-WholeBody) for one cropped person."""

    def __init__(
        self,
        checkpoint: str = "usyd-community/vitpose-plus-base",
        device: str | None = None,
        dataset_index: int = DATASET_COCO,
        joint_indices: dict[str, int] | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("loading %s on %s ...", checkpoint, device)
        self.processor = VitPoseImageProcessor.from_pretrained(checkpoint)
        self.model = (
            VitPoseForPoseEstimation.from_pretrained(checkpoint).to(device).eval()
        )
        self.dataset_index = dataset_index
        self.joint_indices = dict(joint_indices or JOINTS_COCO17)

    @property
    def joint_names(self) -> list[str]:
        return list(self.joint_indices.keys())

    @torch.inference_mode()
    def detect(
        self,
        rgb_bgr: np.ndarray,
        person_bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> Keypoints2D:
        """Run ViTPose on one frame and return the configured joint subset.

        Parameters
        ----------
        rgb_bgr
            ``H×W×3`` uint8 in **BGR** order (the Dataset's native format).
        person_bbox_xyxy
            Person bounding box in image pixels, ``(x1, y1, x2, y2)``.
            Defaults to the full frame.
        """
        if rgb_bgr.dtype != np.uint8 or rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
            raise ValueError(
                f"rgb_bgr must be uint8 H×W×3, got dtype={rgb_bgr.dtype} "
                f"shape={rgb_bgr.shape}"
            )
        H, W = rgb_bgr.shape[:2]
        if person_bbox_xyxy is None:
            person_bbox_xyxy = (0.0, 0.0, float(W), float(H))

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        boxes_per_image = [[list(person_bbox_xyxy)]]  # 1 image, 1 box

        inputs = self.processor(images=rgb, boxes=boxes_per_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        ds_idx = torch.tensor([self.dataset_index], device=self.device)
        outputs = self.model(**inputs, dataset_index=ds_idx)
        post = self.processor.post_process_pose_estimation(
            outputs, boxes=boxes_per_image, threshold=0.0
        )
        # post: list per image, each a list per box, each box dict has
        # 'keypoints' (J_total, 2) and 'scores' (J_total,) tensors.
        result = post[0][0]
        all_kps: torch.Tensor | np.ndarray = result["keypoints"]
        all_scores: torch.Tensor | np.ndarray = result["scores"]
        if hasattr(all_kps, "detach"):
            all_kps = all_kps.detach().cpu().numpy()
            all_scores = all_scores.detach().cpu().numpy()

        joint_names = self.joint_names
        coords = np.stack(
            [np.asarray(all_kps[self.joint_indices[n]], dtype=np.float32)
             for n in joint_names],
            axis=0,
        )
        scores = np.array(
            [float(all_scores[self.joint_indices[n]]) for n in joint_names],
            dtype=np.float32,
        )
        return Keypoints2D(joint_names=joint_names, coords=coords, scores=scores)
