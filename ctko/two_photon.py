from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from gsheets_importer import gsheet2df

TIFF_UMBRELLA = Path("/Volumes/MarcBusche/James/Regular2p")


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def load_data(mouse: str, date: str) -> Tuple[np.ndarray, np.ndarray]:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
    cascade_result = np.load(s2p_path / "cascade_results.npy")
    # cascade_noise = np.load(s2p_path / "noise_levels_cascade.npy")
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    dff = compute_dff(subtract_neuropil(f_raw, f_neu))
    return dff, cascade_result


def process_session(mouse: str, date: str) -> float:
    dff, deconv = load_data(mouse, date)
    nan_frames = np.sum(np.isnan(deconv[0, :]))
    length_seconds = (deconv.shape[1] - nan_frames) / 30
    return np.nansum(deconv, 1) / length_seconds

    # plt.figure()
    # plt.plot(dff[0, :])
    # plt.plot(deconv[0, :])
    # plt.show()


if __name__ == "__main__":

    metadata = gsheet2df("1NZi5kRUMJMPte7jeRmqFrYJIPbUPMDZ_Yq4lWE2ql1k", "Sheet1", 1)
    metadata = metadata[metadata["Suite2p clicked"] == "DONE"]

    rates: Dict[str, float] = {}
    for idx, row in metadata.iterrows():
        mouse = row["Mouse"]
        date = row["Date"]
        print(f"Processing {mouse} {date}")
        try:
            rate = process_session(mouse, date)
            rates[mouse] = rate
        except Exception as e:
            print(f"Error occured for: {mouse} {date}. Error: {e}")

    1 / 0
