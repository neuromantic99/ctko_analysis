from pathlib import Path

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


from ctko.consts import HERE
from ctko.two_photon import load_data
from ctko.utils import moving_average, normalise, threshold_detect

import seaborn as sns

sns.set_theme(context="paper", style="ticks")


def get_correlations(deconv: np.ndarray) -> float:
    matrix = np.corrcoef(deconv[:, ~np.all(np.isnan(deconv), axis=0)])
    off_diagonal_elements = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return np.mean(off_diagonal_elements)


def save_figure(path: Path) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(path, bbox_inches="tight", transparent=True)


def paper_plot(dff: np.ndarray, deconv: np.ndarray, mouse_name: str) -> None:

    plt.figure()
    start = 60
    t = 30 * 120

    n_transients = [len(threshold_detect(cell, 0.2)) for cell in deconv]

    sort_idx = np.argsort(n_transients)

    if mouse_name == "J023":
        cells_plot = [160, 205, sort_idx[3], sort_idx[1], 147]
    elif mouse_name == "J022":
        cells_plot = [35, 20, 23, 44, 55]
    else:
        return

    for n, cell in enumerate(cells_plot):
        flu = moving_average(dff[cell, start : start + t], 10)
        plt.plot(
            (flu - np.min(flu)) + n * 1.5,
            # label=cell,
            color=sns.color_palette()[2],
            label=r"$\Delta$F / F" if n == 0 else None,
        )
        plt.plot(
            deconv[cell, start : start + t] + n * 1.5,
            color=sns.color_palette()[0],
            label="Deconvolved Spikes" if n == 0 else None,
        )

    plt.legend()
    sns.despine()
    plt.xlabel("Time (frames)")
    plt.tight_layout()
    save_figure(HERE.parent / "figures" / f"{mouse_name}_example_neurons.pdf")


def process_session(mouse: str, date: str) -> None:

    dff, deconv, noise_levels, average_movement = load_data(mouse, date)

    # plt.figure()
    paper_plot(dff, deconv, mouse)

    diffed = np.load(HERE.parent / "cache" / f"{mouse}-{date}-diffed.npy")
    movement = moving_average(diffed, 20)
    movement = normalise(movement)

    # Add a value onto the end to get it the same shape as the dff
    movement = np.append(movement, movement[-1])
    # Pupil was not saved for the final recording so this is a valid maneuver for this mouse
    if mouse == "J025" and date == "2024-09-27":
        print("Manual trim")
        dff = dff[:, : len(movement)]
        deconv = deconv[:, : len(movement)]

    assert movement.shape[0] == deconv.shape[1] == dff.shape[1]
    assert deconv.shape[0] == dff.shape[0]

    ## OPTIONAL TO REMOVE NOISE

    # print(noise_levels)
    # deconv = deconv[noise_levels < 2, :]
    # dff = dff[noise_levels < 2, :]
    # deconv[deconv < 0.1] = 0

    resting = movement < 0.1
    n_transients = [len(threshold_detect(cell, 0.2)) for cell in deconv]

    # Resting vs moving correlations
    correlation = get_correlations(deconv)

    print(f"percent resting: {np.sum(resting) / len(resting) * 100} %")
    resting_spike_rate = np.nansum(deconv[:, resting], 1) / (np.sum(resting) / 30)
    moving_spike_rate = np.nansum(deconv[:, ~resting], 1) / (np.sum(~resting) / 30)
    all_spike_rate = np.nansum(deconv, 1) / (len(movement) / 30)

    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-resting-rates.npy", resting_spike_rate
    )

    np.save(HERE.parent / "cache" / f"{mouse}-{date}-n-transients.npy", n_transients)

    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-moving-rates.npy", moving_spike_rate
    )

    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-correlation_coeff.npy",
        correlation,
    )

    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-all-spike-rate.npy", moving_spike_rate
    )

    print(f"Resting spike rate: {np.mean(resting_spike_rate)}")
    print(f"Moving spike rate: {np.mean(moving_spike_rate)}")
    print(f"All spike rate: {np.mean(all_spike_rate)}")


def main() -> None:

    redo = True

    sessions = [
        # ("J022", "2024-09-27"),
        ("J023", "2024-09-27"),
        # ("J025", "2024-09-27"),
        # ("J024", "2024-10-09"),
        # ("J026", "2024-10-24"),
        # ("J027", "2024-10-09"),
        # ("J028", "2024-10-24"),
        # ("J029", "2024-10-25"),
    ]

    for session in sessions:

        mouse, date = session
        save_path = Path(HERE.parent / "cache" / f"{mouse}-{date}-resting-rates.npy")

        if save_path.exists() and not redo:
            print(f"Already processed {mouse} {date}")
            continue

        print(f"Processing {mouse} {date}")
        process_session(mouse, date)
    plt.show()
