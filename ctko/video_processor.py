from pathlib import Path

import numpy as np

from ctko.consts import HERE
from ctko.utils import process_multiple_videos


mouse = "J029"
date = "2024-10-25"


def get_binarise_value(mouse: str) -> int:
    return 100 if mouse in {"J026", "J029"} else 200


def main() -> None:
    pupil_umbrella = Path("/Volumes/MarcBusche/James/Regular2p")
    face_corners = np.load(HERE.parent / "cache" / f"{mouse}_{date}_corner_pixels.npy")

    pupil_folder = pupil_umbrella / date / mouse
    mp4_paths = sorted(list(pupil_folder.glob("*.mp4")))
    diffed = process_multiple_videos(
        mp4_paths, 500, face_corners, get_binarise_value(mouse)
    )
    np.save(HERE.parent / "cache" / f"{mouse}-{date}-diffed.npy", diffed)
