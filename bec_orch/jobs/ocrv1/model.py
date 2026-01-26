"""OCR model wrapper for ONNX inference - GPU only."""

import logging

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.special import log_softmax as scipy_log_softmax

from .data_structures import LineLogits
from .utils import get_execution_providers

logger = logging.getLogger(__name__)

# Check if PyTorch with CUDA is available for GPU operations
_TORCH_CUDA_AVAILABLE = False
_torch = None
_functional = None
try:
    import torch as _torch
    import torch.nn.functional as _functional

    if _torch.cuda.is_available():
        _TORCH_CUDA_AVAILABLE = True
        logger.info("[OCRModel] PyTorch CUDA available - will use GPU for log_softmax and vocab pruning")
    else:
        logger.info("[OCRModel] PyTorch available but CUDA not available - using CPU operations")
except ImportError:
    logger.info("[OCRModel] PyTorch not available - using scipy CPU operations")


class OCRModel:
    def __init__(
        self,
        model_file: str,
        input_layer: str,
        output_layer: str,
        *,
        squeeze_channel: bool,
        swap_hw: bool,
        apply_log_softmax: bool = True,
        use_gpu_operations: bool = True,
        vocab_prune_threshold: float | None = -10.0,
    ) -> None:
        """Initialize OCR model wrapper for ONNX inference.

        Args:
            model_file: Path to ONNX model file
            input_layer: Name of the input layer in the ONNX model
            output_layer: Name of the output layer in the ONNX model
            squeeze_channel: Whether to squeeze channel dimension
            swap_hw: Whether to swap height and width dimensions
            apply_log_softmax: If True, applies log_softmax after forward pass (on GPU or CPU).
                              If False, returns raw logits without softmax transformation.
                              Works for both GPU and CPU execution paths.
            use_gpu_operations: Whether to use GPU for log_softmax and vocab pruning (if available)
            vocab_prune_threshold: Threshold for vocabulary pruning. Tokens with max log_prob
                                   below this threshold are pruned. None disables pruning.
        """
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw
        self._apply_log_softmax = apply_log_softmax
        self._use_gpu = use_gpu_operations and _TORCH_CUDA_AVAILABLE
        self._vocab_prune_threshold = vocab_prune_threshold

        execution_providers = get_execution_providers()
        self.session = ort.InferenceSession(model_file, providers=execution_providers)

        softmax_status = "enabled" if self._apply_log_softmax else "disabled"
        pruning_status = (
            f"enabled (threshold={vocab_prune_threshold})" if vocab_prune_threshold is not None else "disabled"
        )

        if self._use_gpu:
            logger.info(f"[OCRModel] PyTorch GPU: softmax={softmax_status}, vocab_pruning={pruning_status} (per-line)")
        else:
            logger.info(
                f"[OCRModel] scipy/numpy CPU: softmax={softmax_status}, vocab_pruning={pruning_status} (per-line)"
            )

    def _process_logits_gpu(self, logits: npt.NDArray, vocab_axis: int) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Apply log_softmax and vocab pruning on GPU using PyTorch.

        Returns:
            Tuple of (log_probs, keep_indices) where keep_indices is None if no pruning.
        """
        if _torch is None or _functional is None:
            raise RuntimeError("GPU operations requested but PyTorch is not available")

        # Convert to PyTorch tensor on GPU
        logits_tensor = _torch.from_numpy(logits).cuda()

        # Apply log_softmax on vocab axis
        log_probs_tensor = _functional.log_softmax(logits_tensor, dim=vocab_axis)

        # Apply vocabulary pruning on GPU if threshold is set
        keep_indices = None
        if self._vocab_prune_threshold is not None:
            # Find max log prob per vocab token across time
            # For (vocab, time), max over time is dim=1 for 2D, or last dim for batch
            time_axis = 1 if logits.ndim == 2 else 2
            max_per_token = log_probs_tensor.max(dim=time_axis).values

            # For batch, take max across batch too
            if logits.ndim == 3:
                max_per_token = max_per_token.max(dim=0).values

            # Create mask for tokens above threshold (always keep blank at index 0)
            threshold_tensor = _torch.tensor(self._vocab_prune_threshold, device=log_probs_tensor.device)
            keep_mask = max_per_token > threshold_tensor
            keep_mask[0] = True  # Always keep blank token

            # Get indices of kept tokens
            keep_indices = _torch.where(keep_mask)[0].cpu().numpy()

            # Prune vocabulary dimension
            if logits.ndim == 2:  # noqa: SIM108
                # (vocab, time) -> (pruned_vocab, time)
                log_probs_tensor = log_probs_tensor[keep_mask, :]
            else:
                # (batch, vocab, time) -> (batch, pruned_vocab, time)
                log_probs_tensor = log_probs_tensor[:, keep_mask, :]

        # Copy back to CPU numpy array
        log_probs = log_probs_tensor.cpu().numpy().astype(np.float32)

        return log_probs, keep_indices

    def _process_logits_cpu(self, logits: npt.NDArray, vocab_axis: int) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Apply log_softmax and vocab pruning on CPU using scipy/numpy.

        Returns:
            Tuple of (log_probs, keep_indices) where keep_indices is None if no pruning.
        """
        log_probs = scipy_log_softmax(logits, axis=vocab_axis).astype(np.float32)

        # Apply vocabulary pruning on CPU if threshold is set
        keep_indices = None
        if self._vocab_prune_threshold is not None:
            # Find max log prob per vocab token across time
            # For (vocab, time), max over time is axis=1 for 2D
            time_axis = 1 if logits.ndim == 2 else 2
            max_per_token = log_probs.max(axis=time_axis)

            # For batch, take max across batch too
            if logits.ndim == 3:
                max_per_token = max_per_token.max(axis=0)

            # Create mask for tokens above threshold (always keep blank at index 0)
            keep_mask = max_per_token > self._vocab_prune_threshold
            keep_mask[0] = True  # Always keep blank token

            # Get indices of kept tokens
            keep_indices = np.where(keep_mask)[0]

            # Prune vocabulary dimension
            if logits.ndim == 2:  # noqa: SIM108
                # (vocab, time) -> (pruned_vocab, time)
                log_probs = log_probs[keep_mask, :]
            else:
                # (batch, vocab, time) -> (batch, pruned_vocab, time)
                log_probs = log_probs[:, keep_mask, :]

        return log_probs, keep_indices

    def predict(
        self,
        tensor: npt.NDArray,
        content_widths: list[int] | None = None,
        left_pad_widths: list[int] | None = None,
        input_width: int | None = None,
    ) -> list[LineLogits]:
        """Run ONNX inference on preprocessed tensor, return log probabilities.

        Applies log_softmax immediately after inference.
        Also applies per-line vocabulary pruning to reduce IPC bandwidth.

        If content_widths, left_pad_widths and input_width are provided, crops the time dimension
        BEFORE softmax/pruning to remove left and right padding and save computation.

        Args:
            tensor: Input tensor of shape (batch, height, width) or (batch, channels, height, width)
            content_widths: List of content widths (actual text region), one per batch item
            left_pad_widths: List of left padding widths, one per batch item
            input_width: The padded model input width

        Returns:
            List of LineLogits objects containing logits and metadata for each line
        """
        tensor = tensor.astype(np.float32)

        if self._swap_hw:
            tensor = np.transpose(tensor, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            tensor = np.expand_dims(tensor, axis=1)

        ort_value = ort.OrtValue.ortvalue_from_numpy(tensor)
        results = self.session.run_with_ort_values([self._output_layer], {self._input_layer: ort_value})
        logits = np.squeeze(results[0].numpy())

        # Handle single item vs batch
        if logits.ndim == 2:
            # Single item: (vocab, time)
            logits_list = [logits]
            content_widths = content_widths or [logits.shape[1]]
            left_pad_widths = left_pad_widths or [0]
        else:
            # Batch: (batch, vocab, time)
            logits_list = [logits[i] for i in range(logits.shape[0])]
            content_widths = content_widths or [logits.shape[2]] * logits.shape[0]
            left_pad_widths = left_pad_widths or [0] * logits.shape[0]

        # Crop time dimension to remove left and right padding BEFORE softmax/pruning
        # This saves computation by not processing padding frames
        if input_width is not None:
            full_time = logits_list[0].shape[1]  # All items have same full time
            cropped_list = []
            for item_logits, content_w, left_pad_w in zip(logits_list, content_widths, left_pad_widths, strict=True):
                # Calculate start and end timesteps based on pixel positions
                start_timestep = max(0, int(full_time * left_pad_w / input_width))
                end_timestep = max(start_timestep + 1, int(full_time * (left_pad_w + content_w) / input_width))
                cropped_list.append(item_logits[:, start_timestep:end_timestep])
            logits_list = cropped_list

        # Apply log_softmax if enabled
        if self._apply_log_softmax:
            if self._use_gpu:
                processed_list = self._apply_softmax_batch_gpu(logits_list)
            else:
                processed_list = self._apply_softmax_batch_cpu(logits_list)
        else:
            # Keep raw logits
            processed_list = logits_list

        # Apply vocab pruning if threshold is set (independent of softmax)
        if self._vocab_prune_threshold is not None:
            if self._use_gpu:
                processed_list, keep_indices_list = self._prune_batch_gpu(
                    processed_list, logits_list, apply_softmax=self._apply_log_softmax
                )
            else:
                processed_list, keep_indices_list = self._prune_batch_cpu(
                    processed_list, logits_list, apply_softmax=self._apply_log_softmax
                )
        else:
            # No pruning - return processed list (with or without softmax) and None keep_indices
            keep_indices_list = [None] * len(processed_list)

        default_content_width = processed_list[0].shape[1] if processed_list else 0
        content_widths = content_widths or [default_content_width] * len(processed_list)
        left_pad_widths = left_pad_widths or [0] * len(processed_list)

        return [
            LineLogits(
                logits=logits,
                content_width=content_widths[i],
                left_pad_width=left_pad_widths[i],
                keep_indices=keep_indices,
            )
            for i, (logits, keep_indices) in enumerate(zip(processed_list, keep_indices_list, strict=True))
        ]

    def _apply_softmax_batch_gpu(self, logits_list: list[npt.NDArray]) -> list[npt.NDArray]:
        """Apply log_softmax to each item in the batch on GPU.

        Returns:
            List of log_probs arrays (numpy, on CPU)
        """
        if _torch is None or _functional is None:
            raise RuntimeError("GPU operations requested but PyTorch is not available")

        log_probs_list = []
        for logits in logits_list:
            logits_tensor = _torch.from_numpy(logits).cuda()
            log_probs_tensor = _functional.log_softmax(logits_tensor, dim=0)  # vocab axis
            log_probs_list.append(log_probs_tensor.cpu().numpy().astype(np.float32))
        return log_probs_list

    def _prune_batch_gpu(
        self, processed_list: list[npt.NDArray], original_logits_list: list[npt.NDArray], *, apply_softmax: bool
    ) -> tuple[list[npt.NDArray], list[npt.NDArray | None]]:
        """Apply vocabulary pruning per-line on GPU.

        Args:
            processed_list: List of arrays (log_probs if apply_softmax=True, raw logits if False)
            original_logits_list: Original raw logits (used for threshold comparison if softmax disabled)
            apply_softmax: Whether processed_list contains log_probs (True) or raw logits (False)

        Returns:
            Tuple of (list of pruned arrays, list of keep_indices per line)
        """
        if _torch is None or _functional is None:
            raise RuntimeError("GPU operations requested but PyTorch is not available")

        keep_indices_list = []
        pruned_list = []

        for i, processed in enumerate(processed_list):
            # For threshold comparison, use log_probs (compute if needed)
            if apply_softmax:
                # Already have log_probs, use them for threshold
                log_probs_tensor = _torch.from_numpy(processed).cuda()
            else:
                # Need to compute log_softmax temporarily just for threshold comparison
                logits_tensor = _torch.from_numpy(original_logits_list[i]).cuda()
                log_probs_tensor = _functional.log_softmax(logits_tensor, dim=0)

            # Find max log prob per vocab token for this line only
            max_per_token = log_probs_tensor.max(dim=1).values  # max over time

            # Create mask for tokens above threshold
            threshold_tensor = _torch.tensor(self._vocab_prune_threshold, device=log_probs_tensor.device)
            keep_mask = max_per_token > threshold_tensor
            keep_mask[0] = True  # Always keep blank token

            keep_indices = _torch.where(keep_mask)[0].cpu().numpy()
            keep_indices_list.append(keep_indices)

            # Prune the processed array (log_probs or raw logits)
            if apply_softmax:
                pruned_tensor = log_probs_tensor[keep_mask, :]
            else:
                # Prune raw logits using the same mask
                pruned_tensor = _torch.from_numpy(processed).cuda()[keep_mask, :]

            pruned_list.append(pruned_tensor.cpu().numpy().astype(np.float32))

        return pruned_list, keep_indices_list

    def _apply_softmax_batch_cpu(self, logits_list: list[npt.NDArray]) -> list[npt.NDArray]:
        """Apply log_softmax to each item in the batch on CPU.

        Returns:
            List of log_probs arrays
        """
        log_probs_list = []
        for logits in logits_list:
            log_probs = scipy_log_softmax(logits, axis=0).astype(np.float32)
            log_probs_list.append(log_probs)
        return log_probs_list

    def _prune_batch_cpu(
        self, processed_list: list[npt.NDArray], original_logits_list: list[npt.NDArray], *, apply_softmax: bool
    ) -> tuple[list[npt.NDArray], list[npt.NDArray | None]]:
        """Apply vocabulary pruning per-line on CPU.

        Args:
            processed_list: List of arrays (log_probs if apply_softmax=True, raw logits if False)
            original_logits_list: Original raw logits (used for threshold comparison if softmax disabled)
            apply_softmax: Whether processed_list contains log_probs (True) or raw logits (False)

        Returns:
            Tuple of (list of pruned arrays, list of keep_indices per line)
        """
        keep_indices_list = []
        pruned_list = []

        for i, processed in enumerate(processed_list):
            # For threshold comparison, use log_probs (compute if needed)
            if apply_softmax:
                # Already have log_probs, use them for threshold
                log_probs = processed
            else:
                # Need to compute log_softmax temporarily just for threshold comparison
                log_probs = scipy_log_softmax(original_logits_list[i], axis=0).astype(np.float32)

            # Find max log prob per vocab token for this line only
            max_per_token = log_probs.max(axis=1)  # max over time

            # Create mask for tokens above threshold
            keep_mask = max_per_token > self._vocab_prune_threshold
            keep_mask[0] = True  # Always keep blank token

            keep_indices = np.where(keep_mask)[0]
            keep_indices_list.append(keep_indices)

            # Prune the processed array (log_probs or raw logits)
            pruned = processed[keep_mask, :]
            pruned_list.append(pruned)

        return pruned_list, keep_indices_list
