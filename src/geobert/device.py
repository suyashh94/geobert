"""Device detection and management for GPU/CPU fallback."""

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Args:
        prefer_gpu: If True, use GPU when available. If False, always use CPU.

    Returns:
        torch.device: The selected device (cuda, mps, or cpu).
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Apple Silicon GPU support
            return torch.device("mps")

    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.

    Returns:
        dict: Information about CUDA, MPS, and CPU availability.
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device": str(get_device()),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        })

    return info


def print_device_info() -> None:
    """Print device information to stdout."""
    info = get_device_info()
    print("=" * 50)
    print("PyTorch Device Information")
    print("=" * 50)
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"Selected device: {info['device']}")
    print()

    if info["cuda_available"]:
        print("CUDA Information:")
        print(f"  CUDA version: {info['cuda_version']}")
        print(f"  cuDNN version: {info['cudnn_version']}")
        print(f"  GPU count: {info['gpu_count']}")
        print(f"  GPU name: {info['gpu_name']}")
        print(f"  GPU memory: {info['gpu_memory_total']}")
    elif info["mps_available"]:
        print("Apple Silicon MPS is available")
    else:
        print("Running on CPU (no GPU acceleration)")

    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
