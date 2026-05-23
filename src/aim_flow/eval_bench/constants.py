"""Shared defaults for the SPFC evaluation bench."""

from __future__ import annotations

DEFAULT_SEED = 13
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_NUM_INFERENCE_STEPS = 24
DEFAULT_GUIDANCE_SCALE = 4.5

RECTIFIED_CFGPP_REPO_URL = "https://github.com/shreshthsaini/Rectified-CFGpp.git"
RECTIFIED_CFGPP_COMMIT = "3c838882f7b2cdc2e1c3e5785468cdcb1fb27190"

T2I_COMPBENCH_REPO_URL = "https://github.com/Karine-Huang/T2I-CompBench.git"
T2I_COMPBENCH_COMMIT = "1b7094991a57f3c22abdd4f6e8ba6c1a15517073"

T2I_CATEGORY_FILES = {
    "color": "color_val.txt",
    "shape": "shape_val.txt",
    "texture": "texture_val.txt",
    "spatial": "spatial_val.txt",
}

DEFAULT_SPFC_SCHEDULE_1_INDEXED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24]
SPFC_SCHEDULE_ABLATIONS_1_INDEXED = {
    "default": DEFAULT_SPFC_SCHEDULE_1_INDEXED,
    "early_heavy": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "middle_heavy": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "late_heavy": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "uniform": [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24],
}

STEERING_STRENGTH_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
PRIMITIVE_COUNT_SWEEP = [1, 2, 3, 4, 5]

BENCH_METHODS = ["spfc", "rectified_cfgpp", "base"]
