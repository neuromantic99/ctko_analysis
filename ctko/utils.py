import gc
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


def load_video_as_array(mp4_path: Path, chunk_size: int) -> np.ndarray:
    """
    Loads multiple .mp4 files and combines them into a single NumPy array.

    Parameters:
        mp4_paths (list of str): List of file paths to .mp4 files.

    Returns:
        numpy.ndarray: Combined video data in a 4D array
                       (num_videos, num_frames, height, width, channels).
    """

    n_frames = 0

    cap = cv2.VideoCapture(str(mp4_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, 0])

        if n_frames == chunk_size:
            break
        n_frames += 1

    cap.release()

    return np.array(frames)


def diff_tensor(tensor: np.ndarray) -> np.ndarray:
    assert tensor.ndim == 3, "tensor must be 3D"
    return np.abs(np.diff(tensor, axis=0)).sum((1, 2))


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(arr, np.ones(window), "same") / window


def process_video_in_chunks(
    mp4_path: Path, chunk_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a video file in chunks and computes a 1D diffed vector for the entire video.

    Parameters:
        mp4_path (Path): Path to the .mp4 file.
        chunk_size (int): Number of frames to process per chunk.

    Returns:
        numpy.ndarray: The diffed vector for the entire video.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    full_diffed_vector = []
    previous_chunk_last_frame = None  # To store the last frame of the previous chunk
    first_frame = None

    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(
                frame[:, :, 0]
            )  # Extract the grayscale channel (assume it's the first channel)

        if not frames:  # Break the loop if no frames were read
            break

        frames_array = np.array(frames)

        # Binarise the frames
        binarised = (frames_array > 200).astype(int)
        if first_frame is None:
            first_frame = binarised[0, :, :]

        # If there was a previous chunk, include the last frame for continuity in diff computation
        if previous_chunk_last_frame is not None:
            binarised = np.vstack(
                [previous_chunk_last_frame[np.newaxis, ...], binarised]
            )

        # Compute the diffed vector
        diffed = diff_tensor(binarised)
        full_diffed_vector.extend(diffed)

        # Store the last frame of this chunk for continuity in the next iteration
        previous_chunk_last_frame = binarised[-1]

    cap.release()

    return np.array(full_diffed_vector), first_frame, previous_chunk_last_frame


def process_multiple_videos(mp4_paths: list[Path], chunk_size: int) -> np.ndarray:
    """
    Processes multiple video files in chunks and computes a single 1D diffed vector
    for all videos, ensuring continuity between consecutive files.

    Parameters:
        mp4_paths (list of Path): List of paths to the .mp4 files.
        chunk_size (int): Number of frames to process per chunk.

    Returns:
        numpy.ndarray: The concatenated diffed vector for all videos.
    """
    full_diffed_vector: list[int] = []
    previous_video_last_frame = None

    for mp4_path in mp4_paths:
        gc.collect()

        print(f"Processing {mp4_path}")
        diffed, first_frame, last_frame = process_video_in_chunks(mp4_path, chunk_size)

        # If there's a previous video, compute diff between last frame of previous and first of current
        if previous_video_last_frame is not None:
            # Binarise the first frame of the current video
            # Compute diff between last frame of the previous video and first frame of the current
            inter_diff = diff_tensor(
                np.vstack(
                    [
                        previous_video_last_frame[np.newaxis, ...],
                        first_frame[np.newaxis, ...],
                    ]
                )
            )
            full_diffed_vector.extend(inter_diff)

        # Append the current video's diffed vector
        full_diffed_vector.extend(diffed)
        previous_video_last_frame = last_frame

    return np.array(full_diffed_vector)


def array_seconds_to_minute_second(seconds: np.ndarray) -> np.ndarray:
    # Vectorized conversion for efficiency
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return np.array(
        [f"{int(m)}:{int(s):02}" for m, s in zip(minutes, remaining_seconds)], dtype=str
    )
