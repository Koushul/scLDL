import torch


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(8, device="cuda")
        _ = (x + 1).sum().item()
        return True
    except Exception:
        return False


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    if _cuda_usable():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
