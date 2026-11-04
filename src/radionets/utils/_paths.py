def _validate_pre_model_path(train_config):
    checkpoint = train_config.paths.checkpoint
    if not checkpoint:
        raise ValueError(
            f"'pre_model' path is {checkpoint} "
            "even though testing mode was started. Please make sure "
            "you provide a valid path to a model checkpoint file (.ckpt) "
            "in your configuration."
        )
    if not checkpoint.is_file():
        raise ValueError(
            f"'pre_model' path is {checkpoint}, "
            "but not a valid path to a model checkpoint file (.ckpt)."
        )
