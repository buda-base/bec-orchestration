"""
Memory monitoring for pipeline diagnostics.

Provides real-time visibility into system RAM and GPU memory usage
during pipeline execution to help diagnose OOM issues.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("bec.memory")


def _get_process_memory() -> Dict[str, float]:
    """
    Get current process memory usage.
    
    Returns:
        Dict with rss_mb, vms_mb (or empty dict if psutil unavailable)
    """
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
        }
    except ImportError:
        return {}
    except Exception:
        return {}


def _get_system_memory() -> Dict[str, float]:
    """
    Get system-wide memory info.
    
    Returns:
        Dict with total_mb, available_mb, percent_used
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "percent_used": mem.percent,
        }
    except ImportError:
        return {}
    except Exception:
        return {}


def _get_gpu_memory() -> Dict[str, float]:
    """
    Get GPU memory usage (NVIDIA only via pynvml or torch).
    
    Returns:
        Dict with gpu_used_mb, gpu_total_mb, gpu_percent (or empty if unavailable)
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory for default device
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            # Try to get total GPU memory
            try:
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / (1024 * 1024)
                return {
                    "gpu_allocated_mb": allocated,
                    "gpu_reserved_mb": reserved,
                    "gpu_total_mb": total,
                    "gpu_percent": (reserved / total) * 100 if total > 0 else 0,
                }
            except Exception:
                return {
                    "gpu_allocated_mb": allocated,
                    "gpu_reserved_mb": reserved,
                }
    except ImportError:
        pass
    except Exception:
        pass
    
    return {}


def get_memory_snapshot() -> Dict[str, Any]:
    """
    Get a complete memory snapshot (process, system, GPU).
    
    Returns:
        Dict with all available memory metrics
    """
    snapshot = {
        "ts": time.time(),
    }
    snapshot.update(_get_process_memory())
    snapshot.update(_get_system_memory())
    snapshot.update(_get_gpu_memory())
    return snapshot


def log_memory_snapshot(prefix: str = "", level: int = logging.INFO) -> Dict[str, Any]:
    """
    Log current memory state and return the snapshot.
    
    Args:
        prefix: Optional prefix for the log message
        level: Logging level (default INFO)
    
    Returns:
        The memory snapshot dict
    """
    snapshot = get_memory_snapshot()
    
    # Format a human-readable message
    parts = []
    if "rss_mb" in snapshot:
        parts.append(f"RSS={snapshot['rss_mb']:.0f}MB")
    if "available_mb" in snapshot:
        parts.append(f"avail={snapshot['available_mb']:.0f}MB")
    if "percent_used" in snapshot:
        parts.append(f"sys={snapshot['percent_used']:.1f}%")
    if "gpu_reserved_mb" in snapshot:
        parts.append(f"GPU={snapshot['gpu_reserved_mb']:.0f}MB")
    
    msg = ", ".join(parts)
    if prefix:
        msg = f"{prefix}: {msg}"
    
    logger.log(level, msg, extra={"memory_snapshot": snapshot})
    return snapshot


class MemoryMonitor:
    """
    Background async task that periodically logs memory usage.
    
    Usage:
        monitor = MemoryMonitor(interval_s=5.0, queues={"prefetch": q1, "decode": q2})
        monitor_task = asyncio.create_task(monitor.run())
        # ... run pipeline ...
        monitor_task.cancel()
    """
    
    def __init__(
        self,
        interval_s: float = 5.0,
        queues: Optional[Dict[str, asyncio.Queue]] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize memory monitor.
        
        Args:
            interval_s: Seconds between memory checks
            queues: Dict of queue_name -> asyncio.Queue to monitor depths
            log_level: Logging level for periodic logs
        """
        self.interval_s = interval_s
        self.queues = queues or {}
        self.log_level = log_level
        
        # Tracking state
        self._start_time: float = 0
        self._peak_rss_mb: float = 0
        self._peak_rss_time: float = 0
        self._initial_rss_mb: float = 0
        self._sample_count: int = 0
    
    def _get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths."""
        return {name: q.qsize() for name, q in self.queues.items()}
    
    async def run(self) -> None:
        """
        Main loop: log memory every interval_s seconds.
        
        Runs until cancelled.
        """
        self._start_time = time.time()
        self._sample_count = 0
        
        # Get initial memory baseline
        initial = get_memory_snapshot()
        self._initial_rss_mb = initial.get("rss_mb", 0)
        self._peak_rss_mb = self._initial_rss_mb
        self._peak_rss_time = 0
        
        logger.info(
            f"[MemoryMonitor] Started: RSS={self._initial_rss_mb:.0f}MB, "
            f"interval={self.interval_s}s, queues={list(self.queues.keys())}"
        )
        
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                self._sample_count += 1
                
                elapsed = time.time() - self._start_time
                snapshot = get_memory_snapshot()
                queue_depths = self._get_queue_depths()
                
                rss_mb = snapshot.get("rss_mb", 0)
                
                # Track peak
                if rss_mb > self._peak_rss_mb:
                    self._peak_rss_mb = rss_mb
                    self._peak_rss_time = elapsed
                
                # Calculate delta from initial
                delta_mb = rss_mb - self._initial_rss_mb
                delta_sign = "+" if delta_mb >= 0 else ""
                
                # Format queue depths
                q_parts = [f"{name}={depth}" for name, depth in queue_depths.items()]
                q_str = ", ".join(q_parts) if q_parts else "no queues"
                
                # Format memory message
                parts = [f"RSS={rss_mb:.0f}MB ({delta_sign}{delta_mb:.0f}MB)"]
                
                if "available_mb" in snapshot:
                    parts.append(f"avail={snapshot['available_mb']:.0f}MB")
                
                if "gpu_reserved_mb" in snapshot:
                    parts.append(f"GPU={snapshot['gpu_reserved_mb']:.0f}MB")
                
                mem_str = ", ".join(parts)
                
                logger.log(
                    self.log_level,
                    f"[MemoryMonitor] {mem_str} | {q_str} ({elapsed:.0f}s)",
                    extra={
                        "memory": snapshot,
                        "queues": queue_depths,
                        "elapsed_s": elapsed,
                        "delta_mb": delta_mb,
                    }
                )
                
                # Warn if memory growth is concerning (>500MB since start)
                if delta_mb > 500 and self._sample_count % 6 == 0:  # Every 30s at 5s interval
                    logger.warning(
                        f"[MemoryMonitor] High memory growth: {delta_sign}{delta_mb:.0f}MB since start"
                    )
        
        except asyncio.CancelledError:
            # Log final summary
            elapsed = time.time() - self._start_time
            final = get_memory_snapshot()
            final_rss = final.get("rss_mb", 0)
            
            logger.info(
                f"[MemoryMonitor] Stopped after {elapsed:.1f}s, {self._sample_count} samples. "
                f"Final RSS={final_rss:.0f}MB, Peak={self._peak_rss_mb:.0f}MB at t={self._peak_rss_time:.0f}s, "
                f"Growth={final_rss - self._initial_rss_mb:+.0f}MB"
            )
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics (can be called anytime).
        
        Returns:
            Dict with initial, peak, current RSS and timing info
        """
        current = get_memory_snapshot()
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        return {
            "initial_rss_mb": self._initial_rss_mb,
            "peak_rss_mb": self._peak_rss_mb,
            "peak_rss_time_s": self._peak_rss_time,
            "current_rss_mb": current.get("rss_mb", 0),
            "elapsed_s": elapsed,
            "sample_count": self._sample_count,
            "growth_mb": current.get("rss_mb", 0) - self._initial_rss_mb,
        }
