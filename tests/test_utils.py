from typing import Generator, List, Callable
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from ctko.utils import process_multiple_videos, process_video_in_chunks

# Import your function here
# from your_module import process_video_in_chunks


def triplicate_array(a: np.ndarray) -> np.ndarray:
    return np.repeat(a[:, :, np.newaxis], 3, axis=2)


def test_triplicate_array() -> None:
    a = np.array([[1, 2], [3, 4]])
    print(a)
    triplicated = triplicate_array(a)
    assert np.array_equal(
        triplicated, np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    )
    assert triplicated.shape == (2, 2, 3)
    assert np.array_equal(triplicated[:, :, 0], triplicated[:, :, 1])
    assert np.array_equal(triplicated[:, :, 1], triplicated[:, :, 2])


@pytest.fixture
def mock_video_frames_simple() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([150, 150, 150, 150]).reshape(2, 2)
    frame2 = np.array([250, 250, 250, 250]).reshape(2, 2)
    frame3 = np.array([150, 150, 150, 150]).reshape(2, 2)

    frames = [
        (True, triplicate_array(frame1)),
        (True, triplicate_array(frame2)),
        (True, triplicate_array(frame3)),
    ]
    return frames


@pytest.fixture
def mock_video_frames_three_pixel_changes() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([150, 150, 150, 150]).reshape(2, 2)
    frame2 = np.array([250, 100, 250, 250]).reshape(2, 2)
    frames = [
        (True, triplicate_array(frame1)),  # Frame 1
        (True, triplicate_array(frame2)),  # Frame 1
    ]
    return frames


@pytest.fixture
def mock_video_frames_multiple_chunks() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([150, 150, 150, 150]).reshape(2, 2)
    frame2 = np.array([250, 100, 250, 250]).reshape(2, 2)
    frames = [
        (True, triplicate_array(frame1)),  # Frame 1
        (True, triplicate_array(frame2)),  # Frame 1
    ] * 100
    return frames


def test_process_video_in_chunks(mock_video_frames_simple: List) -> None:
    """
    Test the process_video_in_chunks function by mocking cv2.VideoCapture.
    """

    def mock_read() -> Generator:
        for frame in mock_video_frames_simple:
            yield frame
        while True:
            yield (False, None)  # Simulate end of video

    mock_cap = MagicMock()
    mock_cap.read = MagicMock(side_effect=mock_read())

    with patch("cv2.VideoCapture", return_value=mock_cap):
        video_path = Path("dummy_path.mp4")
        result, _, _ = process_video_in_chunks(video_path, 2)

    expected = [4, -4]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_process_video_in_chunks2(mock_video_frames_three_pixel_changes: List) -> None:
    """
    Test the process_video_in_chunks function by mocking cv2.VideoCapture.
    """

    def mock_read() -> Generator:
        for frame in mock_video_frames_three_pixel_changes:
            yield frame
        while True:
            yield (False, None)  # Simulate end of video

    mock_cap = MagicMock()
    mock_cap.read = MagicMock(side_effect=mock_read())

    with patch("cv2.VideoCapture", return_value=mock_cap):
        video_path = Path("dummy_path.mp4")
        chunk_size = 1
        result, _, _ = process_video_in_chunks(video_path, chunk_size)

    expected = [3]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_process_video_in_chunks_multple_chunks(
    mock_video_frames_multiple_chunks: List,
) -> None:
    """
    Test the process_video_in_chunks function by mocking cv2.VideoCapture.
    """
    # ensure that the result is the same regardless of the chunk size
    # Even if the chunk size is longer than the video
    for chunk_size in range(1, 120):

        def mock_read() -> Generator:

            for frame in mock_video_frames_multiple_chunks:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        mock_cap = MagicMock()
        mock_cap.read = MagicMock(side_effect=mock_read())

        with patch("cv2.VideoCapture", return_value=mock_cap):
            video_path = Path("dummy_path.mp4")
            result, _, _ = process_video_in_chunks(video_path, chunk_size)
            assert len(result) == 199

        # Take the last one off as it's diffed
        expected = ([3, -3] * 100)[:-1]
        assert np.array_equal(
            result, expected
        ), f"Expected {expected}, but got {result}"


import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

# Import your functions here
# from your_module import process_video_in_chunks, process_multiple_videos


@pytest.fixture
def mock_video_frames_1() -> List:
    """
    Mock frames for the first video.
    """
    frames = [
        (True, np.full((10, 10, 3), 150, dtype=int)),  # Frame 1
        (True, np.full((10, 10, 3), 250, dtype=int)),  # Frame 2
        (False, None),  # End of video
    ]
    return frames


@pytest.fixture
def mock_video_frames_2() -> List:
    """
    Mock frames for the second video.
    """
    frames = [
        (True, np.full((10, 10, 3), 250, dtype=int)),  # Frame 1
        (True, np.full((10, 10, 3), 50, dtype=int)),  # Frame 2
        (False, None),  # End of video
    ]
    return frames


def test_process_multiple_videos(
    mock_video_frames_1: List, mock_video_frames_2: List
) -> None:
    """
    Test the process_multiple_videos function by mocking cv2.VideoCapture.
    """

    # Mocking two video files
    def mock_read_1() -> Generator:
        for frame in mock_video_frames_1:
            yield frame
        while True:
            yield (False, None)  # Simulate end of video

    def mock_read_2() -> Generator:
        for frame in mock_video_frames_2:
            yield frame
        while True:
            yield (False, None)  # Simulate end of video

    mock_cap_1 = MagicMock()
    mock_cap_1.read = MagicMock(side_effect=mock_read_1())

    mock_cap_2 = MagicMock()
    mock_cap_2.read = MagicMock(side_effect=mock_read_2())

    with patch("cv2.VideoCapture") as mock_video_capture:
        mock_video_capture.side_effect = [mock_cap_1, mock_cap_2]

        # Define paths (dummy paths for testing)
        video_paths = [Path("video1.mp4"), Path("video2.mp4")]
        chunk_size = 2
        result = process_multiple_videos(video_paths, chunk_size)

    expected = [100, 0, -100]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


@pytest.fixture
def mock_video_frames_multiple_chunks_for_multiple_videos1() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([150, 150, 150, 150]).reshape(2, 2)
    frame2 = np.array([250, 100, 250, 250]).reshape(2, 2)
    frame3 = np.array([250, 250, 250, 250]).reshape(2, 2)
    frames = [
        (True, triplicate_array(frame1)),  # Frame 1
        (True, triplicate_array(frame2)),  # Frame 1
        (True, triplicate_array(frame3)),  # Frame 1
    ] * 100
    return frames


@pytest.fixture
def mock_video_frames_multiple_chunks_for_multiple_videos2() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([150, 150, 150, 150]).reshape(2, 2)
    frame2 = np.array([250, 100, 250, 250]).reshape(2, 2)
    frame3 = np.array([250, 250, 250, 250]).reshape(2, 2)
    frames = [
        (True, triplicate_array(frame1)),  # Frame 1
        (True, triplicate_array(frame2)),  # Frame 1
        (True, triplicate_array(frame3)),  # Frame 1
    ] * 100
    return frames


def test_process_multiple_videos_long_chunks(
    mock_video_frames_multiple_chunks_for_multiple_videos1: List,
    mock_video_frames_multiple_chunks_for_multiple_videos2: List,
) -> None:
    """
    Test the process_multiple_videos function by mocking cv2.VideoCapture.
    """

    for chunk_size in range(1, 120):

        # Mocking two video files
        def mock_read_1() -> Generator:
            for frame in mock_video_frames_multiple_chunks_for_multiple_videos1:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        def mock_read_2() -> Generator:
            for frame in mock_video_frames_multiple_chunks_for_multiple_videos2:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        mock_cap_1 = MagicMock()
        mock_cap_1.read = MagicMock(side_effect=mock_read_1())

        mock_cap_2 = MagicMock()
        mock_cap_2.read = MagicMock(side_effect=mock_read_2())

        with patch("cv2.VideoCapture") as mock_video_capture:
            mock_video_capture.side_effect = [mock_cap_1, mock_cap_2]

            # Define paths (dummy paths for testing)
            video_paths = [Path("video1.mp4"), Path("video2.mp4")]
            result = process_multiple_videos(video_paths, chunk_size)

        # Take the last one off as it's diffed so you don't cycle back to the first movie again
        expected = ([3, 1, -4, 3, 1, -4] * 100)[:-1]
        assert np.array_equal(
            result, expected
        ), f"Expected {expected}, but got {result}"


@pytest.fixture
def mock_video_frames_multiple_chunks_for_multiple_videos3() -> List:
    """
    Create a mock video sequence for testing.
    Returns a list of tuples (ret, frame).
    """
    frame1 = np.array([300, 150, 150, 150]).reshape(2, 2)
    frames = [
        (True, triplicate_array(frame1)),  # Frame 1
    ] * 100
    return frames


def test_process_multiple_videos_long_chunks_three_videos(
    mock_video_frames_multiple_chunks_for_multiple_videos1: List,
    mock_video_frames_multiple_chunks_for_multiple_videos2: List,
    mock_video_frames_multiple_chunks_for_multiple_videos3: List,
) -> None:
    """
    Test the process_multiple_videos function by mocking cv2.VideoCapture.
    """

    for chunk_size in range(1, 120):

        # Mocking two video files
        def mock_read_1() -> Generator:
            for frame in mock_video_frames_multiple_chunks_for_multiple_videos1:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        def mock_read_2() -> Generator:
            for frame in mock_video_frames_multiple_chunks_for_multiple_videos2:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        def mock_read_3() -> Generator:
            for frame in mock_video_frames_multiple_chunks_for_multiple_videos3:
                yield frame
            while True:
                yield (False, None)  # Simulate end of video

        mock_cap_1 = MagicMock()
        mock_cap_1.read = MagicMock(side_effect=mock_read_1())

        mock_cap_2 = MagicMock()
        mock_cap_2.read = MagicMock(side_effect=mock_read_2())

        mock_cap_3 = MagicMock()
        mock_cap_3.read = MagicMock(side_effect=mock_read_3())

        with patch("cv2.VideoCapture") as mock_video_capture:
            mock_video_capture.side_effect = [mock_cap_1, mock_cap_2, mock_cap_3]

            video_paths = [Path("video1.mp4"), Path("video2.mp4"), Path("video3.mp4")]
            result = process_multiple_videos(video_paths, chunk_size)

        # https://tinyurl.com/4wmzr5r2
        expected = np.array(
            ([3, 1, -4] * 100 + [3, 1, -4] * 99 + [3, 1, -3] + [0] * 100)[:-1]
        )
        assert np.array_equal(
            result, expected
        ), f"Expected {expected}, but got {result}"
