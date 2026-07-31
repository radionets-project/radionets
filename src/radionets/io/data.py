from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import webdataset as wds
from lightning import LightningDataModule
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset


class H5DataSet(Dataset):
    def __init__(self, data_dir, tar_fourier, mode="train"):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        bundle_paths = data_dir.glob(f"samp_{mode}_*.h5")
        self.bundles = natsorted(bundle_paths)
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier

        if not self.bundles:
            raise ValueError("No bundles found! Please check the names of your files.")

    def __call__(self):
        return print("This is the H5DataSet class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        x = self.open_image("x", i)
        y = self.open_image("y", i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])

        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)

        indices, _ = torch.sort(i)
        bundle = torch.div(indices, self.num_img, rounding_mode="floor")
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)

        bundle_paths = [
            h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
        ]
        bundle_paths_str = list(map(str, bundle_paths))

        data = torch.from_numpy(
            np.array(
                [
                    bund[var][img]
                    for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                    for img in image[
                        bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                    ]
                ]
            )
        )

        if self.tar_fourier is False and data.shape[1] == 2:
            raise ValueError(
                "Two channeled data is used despite Fourier being False.\
                    Set Fourier to True!"
            )

        if data.shape[0] == 1:
            data = data.squeeze(0)
        return data.float()


class H5DataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling visibility
    data from HDF5 files.

    This DataModule manages the loading and preparation
    of the visibility datasets for training, validation,
    testing, and prediction stages of radionets.

    Parameters
    ----------
    data_dir : str or :class:`pathlib.Path`
        Directory path containing the HDF5 data files.
    batch_size : int, optional
        Number of samples per batch.
        Default: ``32``
    fourier : bool, optional
        Whether to use Fourier space targets.
        Default: ``False``
    num_workers : int, optional
        Number of worker processes for data loading.
        Default ``10``

    Attributes
    ----------
    fourier : bool
        Flag indicating whether Fourier space inputs/targets
        are used.
    data_dir : str or Path
        Directory path to the data.
    batch_size : int
        Batch size for data loaders.
    num_workers : int
        Number of worker processes.
    vis_train : H5DataSet
        Training dataset used in the Trainer.fit stage.
    vis_val : H5DataSet
        Validation dataset used in the Trainer.validate
        stage.
    vis_test : H5DataSet
        Test dataset used in the Trainer.test stage.
    vis_predict : H5DataSet
        Prediction dataset used for inference in the
        Trainer.predict stage.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        batch_size: int = 32,
        fourier: bool = False,
        num_workers: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.fourier = fourier
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_length = None
        self.valid_length = None
        self.test_length = None
        self.predict_length = None

        self.save_hyperparameters()
        self.setup("fit")

    def setup(self, stage: str):
        """
        Set up datasets for the specified stage.

        Creates H5DataSet instances for the appropriate data
        splits based on the stage of the workflow
        (fit, test, or predict).

        Parameters
        ----------
        stage : str
            The stage of the workflow. Must be one of:

            - 'fit': Prepares training and validation datasets
            - 'test': Prepares test dataset
            - 'predict': Prepares prediction dataset

        Raises
        ------
        ValueError
            If the provided stage is not one of 'fit', 'test',
            or 'predict'.

        Notes
        -----
        This method is called automatically by PyTorch Lightning
        before any one of the training, validation, testing, or
        prediction loop begins.
        """
        match stage:
            case "fit":
                self.vis_train = H5DataSet(
                    self.data_dir,
                    tar_fourier=self.fourier,
                    mode="train",
                )
                self.vis_val = H5DataSet(
                    self.data_dir,
                    tar_fourier=self.fourier,
                    mode="valid",
                )
                self.train_length = len(self.vis_train)
                self.valid_length = len(self.vis_val)

            case "test":
                self.vis_test = H5DataSet(
                    self.data_dir,
                    tar_fourier=self.fourier,
                    mode="test",
                )
                self.test_length = len(self.vis_test)
            case "predict":
                self.vis_predict = H5DataSet(
                    self.data_dir,
                    tar_fourier=self.fourier,
                    mode="test",
                )
                # NOTE: For now, this will look for test files,
                # but in the future this stage should be used for
                # inference only
                self.predict_length = len(self.vis_predict)
            case _:
                raise ValueError(
                    f"Stage {stage} is not available in {self.__class__.__name__}"
                )

    def train_dataloader(self):
        """
        Create and return the training DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the training dataset with
            configured batch size and number of workers.
        """
        return DataLoader(
            self.vis_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create and return the validation DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the validation dataset
            with configured batch size and number of workers.
        """
        return DataLoader(
            self.vis_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Create and return the test DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the test dataset with
            configured batch size and number of workers.
        """
        return DataLoader(
            self.vis_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """
        Create and return the prediction DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the prediction dataset
            with configured batch size and number of workers.
        """
        return DataLoader(
            self.vis_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def identity(x):
    """Identity function for no-op transformations."""
    return x


class WebDatasetModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling visibility
    data from WebDataset files.

    This DataModule manages the loading and preparation
    of the visibility datasets for training, validation,
    testing, and prediction stages of radionets.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing WebDataset tar files.
        Expected structure:
        - train-{000000..NNNNN}.tar
        - valid-{000000..NNNNN}.tar
        - test-{000000..NNNNN}.tar
    epochs : int
        Number of epochs. Used for WebDataset.with_epoch().
    batch_size : int, optional
        Number of samples per batch. Default: 32
    fourier : bool, optional
        Whether inputs/targets are in Fourier space. Default: False
    num_workers : int, optional
        Number of worker processes. Default: 10
    prefetch_factor : int, optional
        Number of batches to prefetch per worker. Default: 2
    persistent_workers : bool, optional
        Keep workers alive between epochs. Default: True
    transform : Callable, optional
        Transform applied to inputs. Default: None
    target_transform : Callable, optional
        Transform applied to targets. Default: None
    shuffle_buffer : int, optional
        Size of shuffle buffer for training. Default: None

    Notes
    -----
    WebDataset files should be created with the following structure:
    - __key__: unique identifier
    - *.input.npy: input visibility data as numpy array in binary file format
    - *.target.npy: target image data as numpy array in binary file format
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        batch_size: int = 32,
        fourier: bool = False,
        num_workers: int = 10,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        shuffle_buffer: int | None = None,
        compressed: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.fourier = fourier
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.transform = transform or identity
        self.target_transform = target_transform or identity
        self.shuffle_buffer = shuffle_buffer
        self.compressed = compressed

        self._get_dataset_lengths()

        self.save_hyperparameters(ignore=["transform", "target_transform"])

    def _get_dataset_lengths(self):
        train_parquet = list(self.data_dir.glob("train-*.parquet"))[0]
        valid_parquet = list(self.data_dir.glob("valid-*.parquet"))[0]
        test_parquet = list(self.data_dir.glob("test-*.parquet"))[0]

        self.train_length = (
            pq.read_table(train_parquet)
            .to_pandas()["total_samples_in_dataset"]
            .values[0]
        )
        self.valid_length = (
            pq.read_table(valid_parquet)
            .to_pandas()["total_samples_in_dataset"]
            .values[0]
        )
        self.test_length = (
            pq.read_table(test_parquet)
            .to_pandas()["total_samples_in_dataset"]
            .values[0]
        )

    def _create_dataset(self, mode: str, shuffle: bool = True):
        """
        Create a WebDataset pipeline for the specified mode.

        Parameters
        ----------
        mode : str
            One of 'train', 'valid', or 'test'.
        shuffle : bool, optional
            Whether to shuffle the data. Default: False

        Returns
        -------
        wds.WebDataset
            Configured WebDataset pipeline.
        """
        suffix = "tar.gz" if self.compressed else "tar"

        urls = sorted(map(str, self.data_dir.glob(f"{mode}-*.{suffix}")))

        if not urls:
            raise ValueError(
                f"No WebDataset shards found for mode '{mode}' in {self.data_dir}. "
                f"Expected pattern: {mode}-{{000000..NNNNN}}.{suffix}"
            )
        if shuffle:
            shuffle = self.batch_size

        if not self.shuffle_buffer:
            self.shuffle_buffer = 10 * self.batch_size

        dataset = (
            wds.WebDataset(urls, shardshuffle=True, nodesplitter=wds.split_by_node)
            .decode()
            .to_tuple("input.npy", "target.npy")
            .map_tuple(
                lambda x: torch.from_numpy(x).float(),
                lambda y: torch.from_numpy(y).float(),
            )
            .map_tuple(self.transform, self.target_transform)
            .batched(self.batch_size)
        )

        return dataset

    def setup(self, stage: str):
        """
        Set up datasets for the specified stage.

        Parameters
        ----------
        stage : str
            One of 'fit', 'test', or 'predict'.
        """
        match stage:
            case "fit":
                self.train_dataset = self._create_dataset("train", shuffle=True)
                self.val_dataset = self._create_dataset("valid", shuffle=False)
            case "test":
                self.test_dataset = self._create_dataset("test", shuffle=False)
            case "predict":
                self.predict_dataset = self._create_dataset("test", shuffle=False)

                predict_parquet = list(self.data_dir.glob("predict-*.parquet"))[0]
                self.predict_length = (
                    pq.read_table(predict_parquet)
                    .to_pandas()["total_samples_in_dataset"]
                    .values[0]
                )
            case _:
                raise ValueError(
                    f"Stage '{stage}' is not available in {self.__class__.__name__}"
                )

    def _create_dataloader(self, dataset):
        """
        Create a WebLoader with shuffling and batching.

        Parameters
        ----------
        dataset : wds.WebDataset
            WebDataset to wrap.

        Returns
        -------
        wds.WebLoader
            WebLoader DataLoader.
        """
        return (
            wds.WebLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=None,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                persistent_workers=self.persistent_workers
                if self.num_workers > 0
                else False,
                pin_memory=True,
            )
            .unbatched()
            .shuffle(self.shuffle_buffer)
            .batched(self.batch_size)
        )

    def train_dataloader(self):
        """Create training DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the training dataset
            with configured batch size and number of workers.
        """
        loader = self._create_dataloader(self.train_dataset)
        return loader.with_epoch(self.train_length // (self.batch_size * 1))

    def val_dataloader(self):
        """Create validation DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the validation dataset
            with configured batch size and number of workers.
        """
        return self._create_dataloader(self.val_dataset).with_epoch(
            self.test_length // (self.batch_size * 1)
        )

    def test_dataloader(self):
        """Create test DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the test dataset
            with configured batch size and number of workers.
        """
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self):
        """Create prediction DataLoader.

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
            PyTorch DataLoader for the prediction dataset
            with configured batch size and number of workers.
        """
        return self._create_dataloader(self.predict_dataset)
