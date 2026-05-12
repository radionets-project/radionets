from pathlib import Path

import lightning as L
import rich_click as click
import torch
from rich.pretty import pretty_repr

from radionets.core.logging import _setup_logger
from radionets.evaluation.utils import _method_factory, apply_symmetry, get_ifft
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

    _method_factory(eval_config)

    if len(eval_config.paths.model_paths) == 2:
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
            eval_config.paths.model_paths[0]
        )
        train_module_ch1 = TrainModule.load_from_checkpoint(
            eval_config.paths.model_paths[1]
        )
        pred_ch0 = (
            trainer_ch0.predict(model=train_module_ch0, datamodule=data_module)
            .detach()
            .cpu()
        )
        pred_ch1 = (
            trainer_ch1.test(model=train_module_ch1, datamodule=data_module)
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
        )

        model_output = trainer.predict(model=train_module, datamodule=data_module)

    if not eval_config.paths.save_path.exists():
        LOGGER.warning(
            f"Target directory ('save_path') '{eval_config.paths.save_path}' "
            "does not exist. Creating directory..."
        )
        eval_config.paths.save_path.mkdir(parents=True)

    # Stack model_output: The output has shape (N, B, T, C, H, W)
    #                                           |  |  |  |  |  |
    # N: Number of batches ---------------------+  |  |  |  |  |
    # B: Images per batch -------------------------+  |  |  |  |
    # T: Channel for prediction [0] or target [1]-----+  |  |  |
    # C: Channel real [0]/imag [1] or amp [0]/phase [1] -+  |  |
    # H: Height --------------------------------------------+  |
    # W: Width ------------------------------------------------+
    model_output: torch.Tensor = torch.stack(model_output, dim=1)  # ty:ignore[invalid-argument-type]
    preds: torch.Tensor = model_output[:, :, 0]
    targets: torch.Tensor = model_output[:, :, 1]

    # Make images symmetrical again and apply ifft to get image
    # space representation
    preds_ifft = get_ifft(apply_symmetry(preds))
    targets_ifft = get_ifft(apply_symmetry(targets))

    preds_ifft = preds_ifft.reshape(-1, *preds_ifft.shape[-2:])
    targets_ifft = targets_ifft.reshape(-1, *targets_ifft.shape[-2:])

    metrics = {}
    for field in eval_config.evaluation:
        if hasattr(field[1], "met_cls"):
            print(field[0], field[1].__class__.__name__)
            metrics[field[0]] = field[1].met_cls.compute()

    print(pretty_repr(metrics))


if __name__ == "__main__":
    main()
