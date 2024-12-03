from pathlib import Path

import numpy as np

from ctko.consts import HERE
from ctko.utils import process_multiple_videos

mouse = "J024"
date = "2024-10-09"


def main() -> None:
    pupil_umbrella = Path("/Volumes/MarcBusche/James/Regular2p")

    pupil_folder = pupil_umbrella / date / mouse
    mp4_paths = sorted(list(pupil_folder.glob("*.mp4")))

    diffed = process_multiple_videos(mp4_paths, 1000)
    np.save(HERE.parent / "cache" / f"{mouse}-{date}-diffed.npy", diffed)
