from pathlib import Path

import lightning as L
import rich_click as click
import torch
from rich.pretty import pretty_repr

from radionets.core.logging import _setup_logger
from radionets.evaluation.utils import apply_symmetry, get_ifft
from radionets.io import InferenceConfig
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

    inference_config = InferenceConfig.from_toml(config_path)
    LOGGER.info(pretty_repr(inference_config))

    data_module = inference_config.dataloader.module(
        data_dir=inference_config.paths.data_path,
        **inference_config.dataloader.model_dump(),
    )  # ty:ignore[call-non-callable]

    if len(inference_config.paths.model_paths) == 2:
        if not inference_config.dataloader.fourier:
            raise RuntimeError("Cannot load two models if fourier is set to False")

        trainer_ch0 = L.Trainer(
            limit_test_batches=data_module.test_length
            // inference_config.dataloader.batch_size
            if data_module.test_length
            else inference_config.dataloader.batch_size,
            devices=inference_config.devices.num_devices[0],
            accelerator=inference_config.devices.accelerator[0],
            precision=inference_config.devices.precision[0],  # ty:ignore[invalid-argument-type]
            strategy=inference_config.devices.deepspeed[0]
            if inference_config.devices.deepspeed[0]
            else "auto",  # ty:ignore[invalid-argument-type]
        )
        trainer_ch1 = L.Trainer(
            limit_test_batches=data_module.test_length
            // inference_config.dataloader.batch_size
            if data_module.test_length
            else inference_config.dataloader.batch_size,
            devices=inference_config.devices.num_devices[1],
            accelerator=inference_config.devices.accelerator[1],
            precision=inference_config.devices.precision[1],  # ty:ignore[invalid-argument-type]
            strategy=inference_config.devices.deepspeed[1]
            if inference_config.devices.deepspeed[1]
            else "auto",  # ty:ignore[invalid-argument-type]
        )

        trainer_ch0.radionets_task = "inference"
        trainer_ch1.radionets_task = "inference"

        train_module_ch0 = TrainModule.load_from_checkpoint(
            inference_config.paths.model_paths[0],
            weights_only=inference_config.model.weights_only[0],
        )
        train_module_ch1 = TrainModule.load_from_checkpoint(
            inference_config.paths.model_paths[1],
            weights_only=inference_config.model.weights_only[1],
        )
        pred_ch0 = (
            trainer_ch0.predict(model=train_module_ch0, datamodule=data_module)
            .detach()
            .cpu()
        )
        pred_ch1 = (
            trainer_ch1.predict(model=train_module_ch1, datamodule=data_module)
            .detach()
            .cpu()
        )

        model_output = torch.cat(
            (pred_ch0[:, 0].unsqueeze(1), pred_ch1[:, 1].unsqueeze(1)),  # ty:ignore[invalid-argument-type]
            dim=1,
        )
    else:
        trainer = L.Trainer(
            limit_test_batches=data_module.test_length
            // inference_config.dataloader.batch_size
            if data_module.test_length
            else inference_config.dataloader.batch_size,
            devices=inference_config.devices.num_devices[0],
            accelerator=inference_config.devices.accelerator[0],
            precision=inference_config.devices.precision[0],  # ty:ignore[invalid-argument-type]
            strategy=inference_config.devices.deepspeed[0]
            if inference_config.devices.deepspeed[0]
            else "auto",  # ty:ignore[invalid-argument-type]
        )

        trainer.radionets_task = "inference"

        train_module = TrainModule.load_from_checkpoint(
            inference_config.paths.model_paths[0],
            eval_methods=inference_config.evaluation,
            weights_only=inference_config.model.weights_only[0],
        )

        model_output = trainer.predict(model=train_module, datamodule=data_module)

    if not inference_config.paths.save_path.exists():
        LOGGER.warning(
            f"Target directory ('save_path') '{inference_config.paths.save_path}' "
            "does not exist. Creating directory..."
        )
        inference_config.paths.save_path.mkdir(parents=True)

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
    preds: torch.Tensor = torch.cat(model_output)  # ty:ignore[invalid-argument-type]

    # Make images symmetrical again and apply ifft to get image
    # space representation
    preds_ifft = get_ifft(apply_symmetry(preds))

    preds_ifft = preds_ifft.reshape(-1, *preds_ifft.shape[-2:])

    split_size = inference_config.save_images.split_size
    if split_size == -1:
        split_size = 1

    preds_split = torch.tensor_split(preds, split_size)
    preds_ifft_split = torch.tensor_split(preds_ifft, split_size)
    for idx, (ps, pis) in enumerate(zip(preds_split, preds_ifft_split)):
        outpath = inference_config.paths.save_path
        if not outpath.is_dir():
            outpath.mkdir(parents=True)

        torch.save(
            obj={"PRED": ps},
            f=outpath / f"pred_{idx}.pt",
        )
        torch.save(
            obj={"PRED": pis},
            f=outpath / f"pred_ifft_{idx}.pt",
        )


if __name__ == "__main__":
    main()
