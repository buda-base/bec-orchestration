import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_execution_providers() -> list[str]:
    """
    Get available ONNX runtime execution providers.

    Prefers CUDA over TensorRT to avoid TensorRT library issues.
    Falls back to CPU if no GPU providers available.

    Returns:
        List of execution provider names to use
    """
    available = ort.get_available_providers()  # ty:ignore[possibly-missing-attribute]
    logger.info("Available ONNX providers: %s", available)

    # Prefer CUDA over TensorRT (TensorRT often has library issues)
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using providers: %s", providers)
        return providers

    # Fallback to CPU
    logger.warning("No GPU providers available, using CPU only")
    return ["CPUExecutionProvider"]
