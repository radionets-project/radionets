from pathlib import Path

import lightning as L
import pandas as pd
import rich_click as click
import torch
from rich.console import Console
from rich.pretty import pretty_repr
from rich.table import Table

from radionets.core.logging import _setup_logger
from radionets.evaluation.utils import _method_factory
from radionets.io import EvalConfig
from radionets.training import TrainModule

LOGGER = _setup_logger(namespace=__name__)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def main(config_path):
    """Starts the radionets training process with
    options specified in configuration file.

    Parameters
    ----------
    configuration_path : str
        Path to the configuration toml file.
    """
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    eval_config = EvalConfig.from_toml(config_path)
    LOGGER.info(pretty_repr(eval_config))

    data_module = eval_config.dataloader.module(
        data_dir=eval_config.paths.data_path,
        **eval_config.dataloader.model_dump(),
    )  # ty:ignore[call-non-callable]

    _method_factory(eval_config.evaluation)

    if len(eval_config.paths.model_paths) == 2:
        if not eval_config.dataloader.fourier:
            raise RuntimeError("Cannot load two models if fourier is set to False")

        trainer_ch0 = L.Trainer(
            limit_test_batches=data_module.test_length
            // eval_config.dataloader.batch_size
            if data_module.test_length
            else eval_config.dataloader.batch_size,
            devices=eval_config.devices.num_devices[0],
            accelerator=eval_config.devices.accelerator[0],
            precision=eval_config.devices.precision[0],  # ty:ignore[invalid-argument-type]
            strategy=eval_config.devices.deepspeed[0]
            if eval_config.devices.deepspeed[0]
            else "auto",  # ty:ignore[invalid-argument-type]
        )
        trainer_ch1 = L.Trainer(
            limit_test_batches=data_module.test_length
            // eval_config.dataloader.batch_size
            if data_module.test_length
            else eval_config.dataloader.batch_size,
            devices=eval_config.devices.num_devices[1],
            accelerator=eval_config.devices.accelerator[1],
            precision=eval_config.devices.precision[1],  # ty:ignore[invalid-argument-type]
            strategy=eval_config.devices.deepspeed[1]
            if eval_config.devices.deepspeed[1]
            else "auto",  # ty:ignore[invalid-argument-type]
        )

        train_module_ch0 = TrainModule.load_from_checkpoint(
            eval_config.paths.model_paths[0],
            weights_only=eval_config.model.weights_only[0],
        )
        train_module_ch1 = TrainModule.load_from_checkpoint(
            eval_config.paths.model_paths[1],
            weights_only=eval_config.model.weights_only[1],
        )
        pred_ch0 = (
            trainer_ch0.predict(model=train_module_ch0, datamodule=data_module)
            .detach()  # ty:ignore[unresolved-attribute]
            .cpu()
        )
        pred_ch1 = (
            trainer_ch1.predict(model=train_module_ch1, datamodule=data_module)
            .detach()  # ty:ignore[unresolved-attribute]
            .cpu()
        )

        model_output = torch.cat(
            (pred_ch0[:, 0].unsqueeze(1), pred_ch1[:, 1].unsqueeze(1)),
            dim=1,
        )
    else:
        trainer = L.Trainer(
            limit_test_batches=data_module.test_length
            // eval_config.dataloader.batch_size
            if data_module.test_length
            else eval_config.dataloader.batch_size,
            devices=eval_config.devices.num_devices[0],
            accelerator=eval_config.devices.accelerator[0],
            precision=eval_config.devices.precision[0],  # ty:ignore[invalid-argument-type]
            strategy=eval_config.devices.deepspeed[0]
            if eval_config.devices.deepspeed[0]
            else "auto",  # ty:ignore[invalid-argument-type]
        )

        train_module = TrainModule.load_from_checkpoint(
            eval_config.paths.model_paths[0],
            eval_methods=eval_config.evaluation,
            weights_only=eval_config.model.weights_only[0],
        )

        model_output = trainer.predict(model=train_module, datamodule=data_module)

    if not eval_config.paths.save_path.exists():
        LOGGER.warning(
            f"Target directory ('save_path') '{eval_config.paths.save_path}' "
            "does not exist. Creating directory..."
        )
        eval_config.paths.save_path.mkdir(parents=True)

    table = Table(
        title=f"Evaluation Summary for {data_module.predict_length} test images"
    )

    table.add_column("Metric", justify="left", style="dark_sea_green4")
    table.add_column("Mean", justify="right")
    table.add_column("Std. Dev.", justify="right")
    table.add_column("Median", justify="right")

    metrics = {}
    for field in eval_config.evaluation:
        if hasattr(field[1], "met_cls"):
            metrics[field[0]] = field[1].met_cls.compute()
            df = pd.DataFrame(field[1].met_cls.compute())
            df.to_csv(eval_config.paths.save_path / f"{field[0]}.csv", index=False)

            table.add_row(
                field[0],
                str(df.iloc[:, 0].mean()),
                str(df.iloc[:, 0].std()),
                str(df.iloc[:, 0].median()),
            )

    console = Console()
    console.print(table)

    if eval_config.evaluation.save_images:
        # use lazy import
        from radionets.evaluation.utils import get_ifft, apply_symmetry  # noqa: I001

        # Concat model_output: The input has the shape (N, B, P, C, H, W)
        #                                               |  |  |  |  |  |
        # N: Number of batches -------------------------+  |  |  |  |  |
        # B: Images per batch -----------------------------+  |  |  |  |
        # P: Channel for prediction [0] or target [1]---------+  |  |  |
        # C: Channel real [0]/imag [1] or amp [0]/phase [1] -----+  |  |
        # H: Height ------------------------------------------------+  |
        # W: Width ----------------------------------------------------+
        #
        # And the output becomes: (N * B, P, C, H, W)
        model_output: torch.Tensor = torch.cat(model_output)  # ty:ignore[invalid-argument-type]

        num_images = eval_config.evaluation.save_images.num_images  # ty:ignore[unresolved-attribute]
        random_sampling = eval_config.evaluation.save_images.random_sampling  # ty:ignore[unresolved-attribute]

        if num_images and random_sampling:
            if isinstance(random_sampling, int):
                torch.manual_seed(random_sampling)

            im_slice = torch.randint(low=0, high=len(model_output), size=(num_images,))
        elif num_images:
            im_slice = slice(num_images)
        else:
            im_slice = slice(None)

        preds: torch.Tensor = model_output[im_slice, 0]
        targets: torch.Tensor = model_output[im_slice, 1]

        # Make images symmetrical again and apply ifft to get image
        # space representation
        preds_ifft = get_ifft(apply_symmetry(preds))
        targets_ifft = get_ifft(apply_symmetry(targets))

        preds_ifft = preds_ifft.reshape(-1, *preds_ifft.shape[-2:])
        targets_ifft = targets_ifft.reshape(-1, *targets_ifft.shape[-2:])

        split_size = eval_config.evaluation.save_images.split_size  # ty:ignore[unresolved-attribute]
        if split_size == -1:
            split_size = 1  # Save all in one file, i.e. 1 split

        preds_split = torch.tensor_split(preds, split_size)
        targets_split = torch.tensor_split(targets, split_size)
        preds_ifft_split = torch.tensor_split(preds_ifft, split_size)
        targets_ifft_split = torch.tensor_split(targets_ifft, split_size)
        for idx, (ps, ts, pis, tis) in enumerate(
            zip(preds_split, targets_split, preds_ifft_split, targets_ifft_split)
        ):
            outpath = eval_config.paths.save_path / "images"
            if not outpath.is_dir():
                outpath.mkdir(parents=True)

            torch.save(
                obj={"PRED": ps, "TARGET": ts},
                f=outpath / f"eval_{idx}.pt",
            )
            torch.save(
                obj={"PRED": pis, "TARGET": tis},
                f=outpath / f"eval_ifft_{idx}.pt",
            )


if __name__ == "__main__":
    main()
