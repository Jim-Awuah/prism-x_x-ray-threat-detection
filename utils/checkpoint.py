"""Checkpoint save / load utilities.
 If training crashes at epoch 40 out of 50, without checkpointing you would lose everything and have to start over. checkpoint.py saves the model's state after every epoch so training can resume from where it left off.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    output_dir: str,
    is_best: bool = False,
) -> None:
    state = {
        "epoch":      epoch,
        "loss":       loss,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    last_path = os.path.join(output_dir, "last.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(output_dir, "best.pth")
        torch.save(state, best_path)
        logger.info("  → saved best checkpoint (loss=%.4f)", loss)


def load_checkpoint(model, path: str) -> int:
    """Load weights into model and return the saved epoch number."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    logger.info("Loaded checkpoint from '%s' (epoch %d)", path, state["epoch"])
    return state["epoch"]