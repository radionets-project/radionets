from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "DeepSpeedConfig",
]


class DeepSpeedConfig(BaseModel):
    """Lightning DeepSpeedStrategy configuration"""

    zero_optimization: bool = True
    stage: int = 2
    remote_device: str | None = None
    offload_optimizer: bool = False
    offload_parameters: bool = False
    offload_params_device: str = "cpu"
    nvme_path: str = "/local_nvme"
    params_buffer_count: int = 5
    params_buffer_size: int = 100_000_000
    max_in_cpu: int = 1_000_000_000
    offload_optimizer_device: str = "cpu"
    optimizer_buffer_count: int = 4
    block_size: int = 1048576
    queue_depth: int = 8
    single_submit: bool = False
    overlap_events: bool = True
    thread_count: int = 1
    pin_memory: bool = False
    sub_group_size: int = 1_000_000_000_000
    contiguous_gradients: bool = True
    overlap_comm: bool = True
    allgather_partitions: bool = True
    reduce_scatter: bool = True
    allgather_bucket_size: int = 200_000_000
    reduce_bucket_size: int = 200_000_000
    zero_allow_untested_optimizer: bool = True
    logging_batch_size_per_gpu: str | int = "auto"
    config: Path | dict | None = None
    logging_level: int = 30
    loss_scale: float = 0
    initial_scale_power: int = 16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: int = 1
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    synchronize_checkpoint_boundary: bool = False
    load_full_weights: bool = False
    exclude_frozen_parameters: bool = False
