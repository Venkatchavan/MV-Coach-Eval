"""MotionSense dataset adapter."""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from mv_coach.core.exceptions import DataLoadError
from mv_coach.data.loader import BaseDataLoader

logger = logging.getLogger(__name__)


class MotionSenseDataLoader(BaseDataLoader):
    """Data loader for MotionSense dataset.

    MotionSense contains accelerometer and gyroscope data from smartphones.
    Activities: downstairs, upstairs, walking, jogging, sitting, standing.
    """

    ACTIVITY_MAP: Dict[str, int] = {
        "dws": 0,  # downstairs
        "ups": 1,  # upstairs
        "wlk": 2,  # walking
        "jog": 3,  # jogging
        "sit": 4,  # sitting
        "std": 5,  # standing
    }

    ACTIVITY_LABELS: Dict[int, str] = {
        0: "downstairs",
        1: "upstairs",
        2: "walking",
        3: "jogging",
        4: "sitting",
        5: "standing",
    }

    def __init__(self, data_dir: str, window_size: int = 128) -> None:
        """Initialize MotionSense data loader.

        Args:
            data_dir: Directory containing MotionSense CSV files.
            window_size: Size of sliding window for time-series segments.
        """
        super().__init__(data_dir)
        self.window_size = window_size
        self.activity_labels = self.ACTIVITY_LABELS
        self.load_data()

    def load_data(self) -> None:
        """Load MotionSense data from CSV files.

        Expected directory structure:
            data_dir/
                A_DeviceMotion_data/
                    subject_1_activity.csv
                    subject_2_activity.csv
                    ...

        Raises:
            DataLoadError: If data files are not found or invalid.
        """
        data_path = Path(self.data_dir) / "A_DeviceMotion_data"

        if not data_path.exists():
            raise DataLoadError(
                f"MotionSense data directory not found: {data_path}\n"
                f"Please download MotionSense dataset and extract to {self.data_dir}"
            )

        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise DataLoadError(f"No CSV files found in {data_path}")

        logger.info(f"Loading MotionSense data from {data_path}")
        logger.info(f"Found {len(csv_files)} CSV files")

        all_windows = []
        all_labels = []
        all_subjects = []

        for csv_file in csv_files:
            try:
                # Parse filename: subject_X_activity.csv
                filename = csv_file.stem
                parts = filename.split("_")
                subject_id = int(parts[1])
                activity_code = parts[2]

                if activity_code not in self.ACTIVITY_MAP:
                    logger.warning(f"Unknown activity code: {activity_code}")
                    continue

                activity_label = self.ACTIVITY_MAP[activity_code]

                # Load CSV
                df = pd.read_csv(csv_file)

                # Select IMU columns (accel + gyro)
                imu_columns = [
                    "userAcceleration.x",
                    "userAcceleration.y",
                    "userAcceleration.z",
                    "rotationRate.x",
                    "rotationRate.y",
                    "rotationRate.z",
                ]

                # Check if all columns exist
                missing_cols = set(imu_columns) - set(df.columns)
                if missing_cols:
                    logger.warning(
                        f"Missing columns in {csv_file}: {missing_cols}"
                    )
                    continue

                data = df[imu_columns].values

                # Create sliding windows
                windows = self._create_windows(data)

                if len(windows) > 0:
                    all_windows.extend(windows)
                    all_labels.extend([activity_label] * len(windows))
                    all_subjects.extend([subject_id] * len(windows))

            except Exception as e:
                logger.warning(f"Error processing {csv_file}: {e}")
                continue

        if len(all_windows) == 0:
            raise DataLoadError("No valid data windows created")

        # Convert to numpy arrays
        self.X = np.array(all_windows, dtype=np.float32)
        self.y = np.array(all_labels, dtype=np.int64)
        self.subject_ids = np.array(all_subjects, dtype=np.int64)

        logger.info(f"Loaded {len(self.X)} windows")
        logger.info(f"Shape: {self.X.shape}")
        logger.info(f"Subjects: {sorted(np.unique(self.subject_ids).tolist())}")
        logger.info(
            f"Activities: {sorted(np.unique(self.y).tolist())}"
        )

    def _create_windows(
        self, data: np.ndarray, stride: int = 64
    ) -> list[np.ndarray]:
        """Create sliding windows from time-series data.

        Args:
            data: Time-series data of shape (n_timesteps, n_features).
            stride: Stride for sliding window.

        Returns:
            List of windows, each of shape (window_size, n_features).
        """
        windows = []
        n_timesteps = len(data)

        for start in range(0, n_timesteps - self.window_size + 1, stride):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)

        return windows
