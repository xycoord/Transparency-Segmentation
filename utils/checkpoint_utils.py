def get_global_step_from_checkpoint(checkpoint_path):
    """
    Returns:
        int: Global training step at which the checkpoint was saved
    Note:
        Checkpoint paths should be in the format "checkpoint-{global_step}".
    """
    return int(checkpoint_path.name.split("-")[1])

def get_latest_checkpoint_path(checkpoint_dir):
    """
    Searches the checkpoint directory for the latest checkpoint.
    Returns:
        Path: Path to the latest checkpoint, or None if no checkpoints
    """
    is_checkpoint_dir = lambda dir: dir.name.startswith("checkpoint")

    checkpoints = list(filter(is_checkpoint_dir, checkpoint_dir.iterdir()))
    if len(checkpoints) == 0:
        return None
    sorted_checkpoints = sorted(checkpoints, key=get_global_step_from_checkpoint)
    latest_checkpoint = sorted_checkpoints[-1]
    return latest_checkpoint 

def get_checkpoint_path(checkpoint_name, checkpoint_dir):
    """
    Returns:
        Path: Path to the specified checkpoint, or None if it does not exist
    """
    if checkpoint_name == "latest":
        return get_latest_checkpoint_path(checkpoint_dir)
    else:
        return checkpoint_dir / checkpoint_name # TODO for consistency, should we check if the file exists? This might require changes to resume_from_checkpoint

def resume_from_checkpoint(checkpoint_name, checkpoint_dir, accelerator, logger):
    """
    Load model and optimizer from a checkpoint to resume training.
    If the checkpoint does not exist, do noting so that the training starts from scratch.

    Returns:
        int: Global training step from checkpoint if loaded, 0 if starting fresh
    """

    checkpoint_path = get_checkpoint_path(checkpoint_name, checkpoint_dir)

    if checkpoint_path is None or not checkpoint_path.exists():
        logger.info(f"Checkpoint '{checkpoint_name}' does not exist. Starting a new training run.")
        global_step = 0
    else: # Load Checkpoint
        accelerator.wait_for_everyone()
        accelerator.load_state(checkpoint_path)
        global_step = get_global_step_from_checkpoint(checkpoint_path)
    return global_step

