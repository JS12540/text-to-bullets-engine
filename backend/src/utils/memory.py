"""Memory and resource tracking utilities."""
import gc
import psutil
import torch


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage stats."""
    process = psutil.Process()
    mem = process.memory_info()

    result = {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
    }

    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

    return result


def force_cleanup() -> None:
    """Force garbage collection and CUDA cache cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
