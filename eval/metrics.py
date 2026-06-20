from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.paths import FILTERED_METADATA_CSV, SAMPLE_IMAGES_DIR, SYNTHETIC_IMAGES_DIR, SYNTHETIC_METADATA_CSV

MISMATCH_THRESHOLD = 0.25
HIGH_VARIANCE_RANGE = 0.15
RUNS_PER_IMAGE = 3
STABILITY_SAMPLE_SIZE = 10


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    ranked = aligned.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def normalize_human(shade_1to5: pd.Series) -> pd.Series:
    return (shade_1to5.astype(float) - 1) / 4


def load_eval_metadata() -> pd.DataFrame:
    real = pd.read_csv(FILTERED_METADATA_CSV)
    real = real.copy()
    real["source"] = "mapillary"

    if SYNTHETIC_METADATA_CSV.exists():
        synthetic = pd.read_csv(SYNTHETIC_METADATA_CSV)
        synthetic = synthetic.copy()
        synthetic["source"] = synthetic.get("source", "synthetic").fillna("synthetic")
        return pd.concat([real, synthetic], ignore_index=True)

    return real


def resolve_image_path(uuid: str) -> Path | None:
    for base in (SAMPLE_IMAGES_DIR, SYNTHETIC_IMAGES_DIR):
        for ext in (".jpeg", ".jpg", ".png"):
            candidate = base / f"{uuid}{ext}"
            if candidate.exists():
                return candidate
    return None


def build_eval_frame(
    labels: pd.DataFrame,
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    labeled = labels.dropna(subset=["shade_1to5"]).copy()
    labeled_uuids = set(labeled["uuid"])
    scored_uuids = set(scores["uuid"])
    meta_uuids = set(metadata["uuid"])

    missing_scores = sorted(labeled_uuids - scored_uuids)
    missing_meta = sorted(labeled_uuids - meta_uuids)
    if missing_scores:
        warnings.append(f"Excluded uuids missing from eval/data/scores.csv: {missing_scores}")
    if missing_meta:
        warnings.append(f"Excluded uuids missing from metadata: {missing_meta}")

    frame = (
        labeled.merge(scores, on="uuid", how="inner")
        .merge(metadata, on="uuid", how="inner", suffixes=("", "_meta"))
    )
    frame["human_norm"] = normalize_human(frame["shade_1to5"])
    frame["err_vlm"] = (frame["pedestrian_shade_score"] - frame["human_norm"]).abs()
    return frame, warnings


def filter_real_images(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["source"] == "mapillary"].copy()


def compute_method_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float | int]:
    n = len(frame)
    if n == 0:
        return {"n": 0, "spearman": float("nan"), "mae": float("nan")}
    spearman = _spearman_corr(frame["human_norm"], frame[score_col])
    mae = (frame[score_col] - frame["human_norm"]).abs().mean()
    return {"n": n, "spearman": float(spearman), "mae": float(mae)}


def flag_mismatches(frame: pd.DataFrame, threshold: float = MISMATCH_THRESHOLD) -> pd.DataFrame:
    flagged = frame[frame["err_vlm"] > threshold].copy()
    if flagged.empty:
        return pd.DataFrame(columns=list(frame.columns))
    return flagged


def summarize_mismatches_by_scene_category(mismatches: pd.DataFrame) -> pd.DataFrame:
    if mismatches.empty:
        return pd.DataFrame(columns=["scene_category", "count"])
    return (
        mismatches.groupby("scene_category")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def per_image_run_stats(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby("uuid")["pedestrian_shade_score"]
    return pd.DataFrame(
        {
            "uuid": grouped.mean().index,
            "score_mean": grouped.mean().values,
            "score_std": grouped.std(ddof=0).values,
            "score_range": (grouped.max() - grouped.min()).values,
            "n_runs": grouped.count().values,
        }
    )


def stability_summary(stats: pd.DataFrame, runs: pd.DataFrame) -> dict[str, float | int]:
    if stats.empty:
        return {
            "n_images": 0,
            "median_std": float("nan"),
            "max_range": float("nan"),
            "pct_high_variance": float("nan"),
            "test_retest_spearman": float("nan"),
        }
    run1 = runs[runs["run_id"] == 1].set_index("uuid")["pedestrian_shade_score"]
    run2 = runs[runs["run_id"] == 2].set_index("uuid")["pedestrian_shade_score"]
    test_retest = _spearman_corr(run1, run2)
    return {
        "n_images": len(stats),
        "median_std": float(stats["score_std"].median()),
        "max_range": float(stats["score_range"].max()),
        "pct_high_variance": float((stats["score_range"] > HIGH_VARIANCE_RANGE).mean() * 100),
        "test_retest_spearman": test_retest,
    }


def merge_run_stability(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        out = frame.copy()
        out["run_std"] = float("nan")
        out["run_range"] = float("nan")
        return out
    merged = frame.merge(stats[["uuid", "score_std", "score_range"]], on="uuid", how="left")
    return merged.rename(columns={"score_std": "run_std", "score_range": "run_range"})
