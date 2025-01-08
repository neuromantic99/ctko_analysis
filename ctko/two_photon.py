from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


from ctko.gsheets_importer import gsheet2df

TIFF_UMBRELLA = Path("/Volumes/MarcBusche/James/Regular2p")


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def compute_dff_with_rolling_mean(f: np.ndarray, N: int) -> np.ndarray:
    """
    Compute ΔF/F using a rolling mean for the fluorescence matrix.

    Parameters:
        f (np.ndarray): Fluorescence matrix (n_cells x n_frames).
        N (int): Window size for the rolling mean.

    Returns:
        np.ndarray: ΔF/F matrix with the same shape as `f`.
    """
    # Compute the rolling mean along the time axis (axis=1)
    flu_mean = uniform_filter1d(f, size=N, axis=1, mode="reflect")
    # Compute ΔF/F
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def load_data(
    mouse: str, date: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"

    ops = np.load(s2p_path / "ops.npy", allow_pickle=True).item()
    average_movement = np.mean(np.abs(ops["xoff"]) + np.abs(ops["yoff"]))

    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)[iscell]
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]

    dff = compute_dff_with_rolling_mean(subtract_neuropil(f_raw, f_neu), 30 * 30)

    # cascade_result = np.load(s2p_path / "cascade_results_running_mean_not_zeroed.npy")
    cascade_result = np.load(s2p_path / "cascade_results_not_zeroed.npy")
    noise_level_cascade = np.load(s2p_path / "noise_levels_cascade_running_mean.npy")

    return dff, cascade_result, noise_level_cascade, average_movement
