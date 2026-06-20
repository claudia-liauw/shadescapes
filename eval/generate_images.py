"""Generate synthetic streetscape images for Tier C gap-fill. Requires GOOGLE_API_KEY."""

from __future__ import annotations

import argparse

import pandas as pd
from google import genai
from google.genai import types

from eval.paths import SYNTHETIC_IMAGES_DIR, SYNTHETIC_METADATA_CSV, SYNTHETIC_PROMPTS_CSV
from src import config

IMAGEN_MODEL = "imagen-4.0-generate-001"

PROMPT_TEMPLATE = """Photorealistic street-level photograph, eye height ~1.5m, Singapore tropical setting.
{prompt}
Walkable concrete path occupies roughly {sidewalk_pct} of the lower frame; path is the visual subject, not the road lane.
Lighting consistent with {hour}:00.
No text overlays, no watermarks, no people."""


def build_image_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        prompt=row["prompt"],
        hour=int(row["hour"]),
        sidewalk_pct=row["sidewalk_pct"],
    )


def generate_one(uuid: str, row: pd.Series, force: bool = False) -> bool:
    out_path = SYNTHETIC_IMAGES_DIR / f"{uuid}.jpeg"
    if out_path.exists() and not force:
        print(f"SKIP {uuid}: already exists")
        return False

    client = genai.Client(api_key=config.get_google_api_key())
    prompt = build_image_prompt(row)
    response = client.models.generate_images(
        model=IMAGEN_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="4:3",
        ),
    )
    image = response.generated_images[0].image
    SYNTHETIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"CREATED {out_path}")
    return True


def append_metadata(uuid: str, row: pd.Series) -> None:
    new_row = {
        "uuid": uuid,
        "source": "synthetic",
        "orig_id": "",
        "lat": 1.30,
        "lon": 103.80,
        "hour": int(row["hour"]),
        "heading": float(row["heading"]),
        "place": f"synthetic_{row['scene_category']}",
        "sidewalk_pct": float(row["sidewalk_pct"]),
    }
    if SYNTHETIC_METADATA_CSV.exists():
        existing = pd.read_csv(SYNTHETIC_METADATA_CSV)
        existing = existing[existing["uuid"] != uuid]
        df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(SYNTHETIC_METADATA_CSV, index=False)


def main(uuid_filter: str | None = None, force: bool = False) -> None:
    prompts = pd.read_csv(SYNTHETIC_PROMPTS_CSV)
    if uuid_filter:
        prompts = prompts[prompts["uuid"] == uuid_filter]
    created = skipped = errors = 0
    for _, row in prompts.iterrows():
        uuid = str(row["uuid"])
        try:
            if generate_one(uuid, row, force=force):
                append_metadata(uuid, row)
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {uuid}: {exc}")
    print(f"Summary: created={created}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", help="Generate a single uuid from synthetic_prompts.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images")
    args = parser.parse_args()
    main(uuid_filter=args.uuid, force=args.force)
