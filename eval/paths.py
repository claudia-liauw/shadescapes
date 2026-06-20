from pathlib import Path

from src import config

PROJECT_ROOT = config.PROJECT_ROOT
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_DATA_DIR = EVAL_DIR / "data"
HUMAN_LABELS_CSV = EVAL_DATA_DIR / "human_labels.csv"
EVAL_SCORES_CSV = EVAL_DATA_DIR / "scores.csv"
STABILITY_SAMPLE_CSV = EVAL_DATA_DIR / "stability_sample.csv"
RUN_VARIANCE_CSV = EVAL_DATA_DIR / "run_variance.csv"
SYNTHETIC_PROMPTS_CSV = EVAL_DATA_DIR / "synthetic_prompts.csv"

FILTERED_METADATA_CSV = config.METADATA_CSV
SYNTHETIC_METADATA_CSV = config.DATA_DIR / "synthetic_streetscapes.csv"

SAMPLE_IMAGES_DIR = config.DATA_DIR / "images" / "sample"
SYNTHETIC_IMAGES_DIR = config.DATA_DIR / "images" / "synthetic"
