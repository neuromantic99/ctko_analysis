from pathlib import Path

import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


from ctko.consts import HERE
from ctko.two_photon import load_data
from ctko.utils import moving_average


def process_session(mouse: str, date: str) -> None:
    diffed = np.load(HERE.parent / "cache" / f"{mouse}-{date}-diffed.npy")
    # trace = moving_average(diffed, 10)
    # plt.plot(trace - min(trace))

    movement = moving_average(zscore(diffed), 10)
    # Add a value onto the end to get it the same shape as the dff
    movement = np.append(movement, movement[-1])

    dff, deconv, noise_levels = load_data(mouse, date)

    # Pupil was not saved for the final recording so this is a valid maneuver for this mouse
    if mouse == "J025" and date == "2024-09-27":
        print("Manual trim")
        dff = dff[:, : len(movement)]
        deconv = deconv[:, : len(movement)]

    assert movement.shape[0] == deconv.shape[1] == dff.shape[1]
    assert deconv.shape[0] == dff.shape[0]

    ## OPTIONAL TO REMOVE NOISE
    deconv = deconv[noise_levels < 2, :]
    dff = dff[noise_levels < 2, :]
    deconv[deconv < 0.1] = 0

    sum_values = np.nansum(deconv, axis=1)
    sort_idx = np.argsort(sum_values)
    idx_plot = sort_idx[-1]

    plt.plot(deconv[idx_plot, :])
    flu = dff[idx_plot, :] - np.min(dff[idx_plot, :])
    plt.plot(flu, alpha=0.5, color="red")
    plt.show()

    resting = movement < -0.5
    print(f"percent resting: {np.sum(resting) / len(resting) * 100} %")

    resting_spike_rate = np.nansum(deconv[:, resting], 1) / (np.sum(resting) / 30)
    moving_spike_rate = np.nansum(deconv[:, ~resting], 1) / (np.sum(~resting) / 30)
    all_spike_rate = np.nansum(deconv, 1) / (len(movement) / 30)

    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-resting-rates.npy", resting_spike_rate
    )
    np.save(
        HERE.parent / "cache" / f"{mouse}-{date}-moving-rates.npy", moving_spike_rate
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
        ("J023", "2024-09-27"),
        ("J022", "2024-09-27"),
        ("J024", "2024-10-09"),
        ("J025", "2024-09-27"),
        ("J026", "2024-10-24"),
        ("J027", "2024-10-09"),
        ("J029", "2024-10-25"),
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
