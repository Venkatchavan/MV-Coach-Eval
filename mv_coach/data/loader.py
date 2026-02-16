"""Base data loader interface with LOSO splitting."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mv_coach.core.exceptions import DataLoadError


class HARDataset(Dataset):
    """PyTorch Dataset for HAR time-series data."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
    ) -> None:
        """Initialize HAR dataset.

        Args:
            X: Time-series feature data of shape (n_samples, n_timesteps, n_features).
            y: Labels of shape (n_samples,).
            subject_ids: Subject IDs of shape (n_samples,).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label).
        """
        return self.X[idx], self.y[idx]


class BaseDataLoader(ABC):
    """Abstract base class for HAR data loaders with LOSO splitting."""

    def __init__(self, data_dir: str) -> None:
        """Initialize data loader.

        Args:
            data_dir: Directory containing dataset files.
        """
        self.data_dir = data_dir
        self.X: np.ndarray
        self.y: np.ndarray
        self.subject_ids: np.ndarray
        self.activity_labels: Dict[int, str]

    @abstractmethod
    def load_data(self) -> None:
        """Load raw data from files.

        Must populate self.X, self.y, self.subject_ids, and self.activity_labels.
        """
        pass

    def get_loso_split(
        self, test_subject: int
    ) -> Tuple[HARDataset, HARDataset]:
        """Create Leave-One-Subject-Out (LOSO) train/test split.

        Args:
            test_subject: Subject ID to use for testing.

        Returns:
            Tuple of (train_dataset, test_dataset).

        Raises:
            DataLoadError: If subject ID is invalid.
        """
        if test_subject not in self.subject_ids:
            raise DataLoadError(f"Subject {test_subject} not found in dataset")

        # Train: all subjects except test_subject
        train_mask = self.subject_ids != test_subject
        X_train = self.X[train_mask]
        y_train = self.y[train_mask]
        subject_ids_train = self.subject_ids[train_mask]

        # Test: only test_subject
        test_mask = self.subject_ids == test_subject
        X_test = self.X[test_mask]
        y_test = self.y[test_mask]
        subject_ids_test = self.subject_ids[test_mask]

        train_dataset = HARDataset(X_train, y_train, subject_ids_train)
        test_dataset = HARDataset(X_test, y_test, subject_ids_test)

        return train_dataset, test_dataset

    def get_all_subjects(self) -> List[int]:
        """Get list of all unique subject IDs.

        Returns:
            List of subject IDs.
        """
        return sorted(np.unique(self.subject_ids).tolist())

    def create_dataloaders(
        self,
        train_dataset: HARDataset,
        test_dataset: HARDataset,
        batch_size: int,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for train and test datasets.

        Args:
            train_dataset: Training dataset.
            test_dataset: Test dataset.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def get_num_classes(self) -> int:
        """Get number of activity classes.

        Returns:
            Number of classes.
        """
        return len(self.activity_labels)

    def get_input_shape(self) -> Tuple[int, int]:
        """Get input shape (n_timesteps, n_features).

        Returns:
            Tuple of (timesteps, features).
        """
        return self.X.shape[1], self.X.shape[2]
