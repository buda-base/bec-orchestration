import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_execution_providers() -> list[tuple[str, dict] | str]:
    """
    Get available ONNX runtime execution providers with deterministic settings.

    Prefers CUDA over TensorRT to avoid TensorRT library issues.
    Falls back to CPU if no GPU providers available.
    
    Enables deterministic mode for CUDA to ensure reproducible results.

    Returns:
        List of execution provider names/configs to use
    """
    available = ort.get_available_providers()
    logger.info("Available ONNX providers: %s", available)

    # Prefer CUDA over TensorRT (TensorRT often has library issues)
    if "CUDAExecutionProvider" in available:
        # Configure CUDA for deterministic execution
        cuda_provider_options = {
            "cudnn_conv_algo_search": "DEFAULT",  # Use default deterministic algorithm
            "do_copy_in_default_stream": True,
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider"
        ]
        logger.info("Using CUDA with deterministic settings: %s", cuda_provider_options)
        return providers

    # Fallback to CPU
    logger.warning("No GPU providers available, using CPU only")
    return ["CPUExecutionProvider"]
