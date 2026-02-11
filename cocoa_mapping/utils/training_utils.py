import torch


def get_device() -> torch.device:
    """Return the best device to be used for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
