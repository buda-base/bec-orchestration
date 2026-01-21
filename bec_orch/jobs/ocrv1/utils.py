import logging
import onnxruntime as ort


def get_execution_providers() -> list[str]:
    """
    Get available ONNX runtime execution providers.

    Returns:
        List of available execution provider names
    """
    available_providers = ort.get_available_providers()
    logger = logging.getLogger(__name__)
    logger.info("Available ONNX providers: %s", available_providers)
    return available_providers

