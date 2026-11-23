import os
import shutil


def prune_checkpoints(output_dir, total_limit, logger=None):
    """Keep at most ``total_limit - 1`` checkpoints so that the next save stays within the limit."""
    if total_limit is None:
        return
    ckpts = sorted((d for d in os.listdir(output_dir) if d.startswith("checkpoint-")), key=lambda x: int(x.split("-")[1]))
    for d in ckpts[: max(0, len(ckpts) - total_limit + 1)]:
        if logger is not None:
            logger.info(f"Removing old checkpoint {d}")
        shutil.rmtree(os.path.join(output_dir, d))


def find_resume_checkpoint(output_dir, resume):
    if resume is None:
        return None
    if resume != "latest":
        return resume if os.path.isabs(resume) else os.path.join(output_dir, os.path.basename(resume))
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted((d for d in os.listdir(output_dir) if d.startswith("checkpoint-")), key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None
