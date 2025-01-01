def get_global_step_from_checkpoint(checkpoint_path):
    return int(checkpoint_path.name.split("-")[1])

def get_latest_checkpoint_path(checkpoint_dir):
    is_checkpoint_dir = lambda dir: dir.name.startswith("checkpoint")

    checkpoints = list(filter(is_checkpoint_dir, checkpoint_dir.iterdir()))
    if len(checkpoints) == 0:
        return None
    sorted_checkpoints = sorted(checkpoints, key=get_global_step_from_checkpoint)
    latest_checkpoint = sorted_checkpoints[-1]
    return latest_checkpoint 

def get_checkpoint_path(checkpoint_name, checkpoint_dir):
    if checkpoint_name == "latest":
        return get_latest_checkpoint_path(checkpoint_dir)
    else:
        return checkpoint_dir / checkpoint_name
