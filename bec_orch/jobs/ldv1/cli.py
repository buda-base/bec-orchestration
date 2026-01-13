import argparse
import asyncio
from collections import deque
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from rich.console import Console
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from .types_common import VolumeTask, ImageTask
from .config import PipelineConfig
from .utils import (
    get_s3_folder_prefix,
    gets3blob,
    get_image_list_and_version_s3,
    _normalize_uri,
    _join_uri,
    _get_local_image_tasks,
)

# Send UI + diagnostics to stderr by default (stdout can be reserved for piping results).
console = Console(stderr=True)


def make_progress_hook(q: asyncio.Queue[Dict[str, Any]]):
    # Called from inside pipeline tasks; must not block.
    def hook(evt: Dict[str, Any]) -> None:
        # Always surface fatal events to stderr immediately, even if the UI queue is full.
        if evt.get("type") == "fatal":
            try:
                stage = evt.get("stage", "unknown")
                err = evt.get("error", "unknown error")
                console.print(f"[bold red]FATAL[/bold red] ({stage}): {err}", highlight=False)
                # Print a compact payload for debugging (e.g. record_summary, URIs)
                extra = {k: v for k, v in evt.items() if k not in ("type",)}
                console.print(extra, highlight=False)
            except Exception:
                pass
        try:
            q.put_nowait(evt)
        except asyncio.QueueFull:
            # Drop events under load; pipeline must win.
            # But for fatal, make a best-effort to deliver it to the UI by evicting one older event.
            if evt.get("type") == "fatal":
                try:
                    _ = q.get_nowait()
                    q.put_nowait(evt)
                except Exception:
                    pass
            return
    return hook


def render_queue_table(worker) -> Table:
    rows = [
        ("prefetch→decode", worker.q_prefetcher_to_decoder),
        ("decode→tile", worker.q_decoder_to_tilebatcher),
        ("post→tile", worker.q_postprocessor_to_tilebatcher),
        ("tile→infer", worker.q_tilebatcher_to_inference),
        ("infer1→post", worker.q_gpu_pass_1_to_post_processor),
        ("infer2→post", worker.q_gpu_pass_2_to_post_processor),
        ("post→writer", worker.q_post_processor_to_writer),
    ]

    t = Table(title="Queue fullness", expand=True)
    t.add_column("Queue")
    t.add_column("Size", justify="right")
    t.add_column("Max", justify="right")
    t.add_column("Pressure", justify="right")

    for name, q in rows:
        size = q.qsize()
        mx = q.maxsize or 0
        pct = (100.0 * size / mx) if mx else 0.0
        t.add_row(name, str(size), str(mx), f"{pct:5.1f}%")

    return t


async def ui_loop(
    *,
    events: asyncio.Queue[Dict[str, Any]],
    worker,
    total: Optional[int],
    show_queues: bool,
) -> None:
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    )

    done_task = progress.add_task("Images persisted", total=total)
    err_task = progress.add_task("Errors", total=None)

    flush_state = "idle"
    last_img = ""
    fatal: Optional[str] = None
    fatal_stage: Optional[str] = None
    fatal_hint: Optional[str] = None
    # Sliding-window throughput (images/s over last 3 seconds)
    rate_window_s = 3.0
    item_times_s: "deque[float]" = deque()

    def _prune(now_s: float) -> None:
        cutoff = now_s - rate_window_s
        while item_times_s and item_times_s[0] < cutoff:
            item_times_s.popleft()

    def render():
        # Compose a single live layout (progress + optional queue panel)
        now_s = time.monotonic()
        _prune(now_s)
        rate = (len(item_times_s) / rate_window_s) if rate_window_s > 0 else 0.0

        grid = Table.grid(expand=True)
        grid.add_row(progress)

        meta = Table.grid(expand=True)
        total_s = str(total) if total is not None else "?"
        meta_line = (
            f"Total: [bold]{total_s}[/bold]    "
            f"Rate(3s): [bold]{rate:.2f} img/s[/bold]    "
            f"Flush: [bold]{flush_state}[/bold]    "
            f"Last: {last_img}"
        )
        if fatal:
            meta_line += f"\n[bold red]FATAL[/bold red] ({fatal_stage or 'unknown'}): {fatal}"
            if fatal_hint:
                meta_line += f"\nHint: {fatal_hint}"
        meta.add_row(meta_line)
        grid.add_row(meta)

        if show_queues:
            grid.add_row(render_queue_table(worker))
        return grid

    with Live(render(), refresh_per_second=10, console=console) as live:
        while True:
            try:
                evt = await asyncio.wait_for(events.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # periodic refresh for queue fullness even if no events
                live.update(render())
                continue

            et = evt.get("type")
            if et == "item":
                item_times_s.append(time.monotonic())
                last_img = evt.get("img") or ""
                if evt.get("ok"):
                    progress.advance(done_task, 1)
                else:
                    progress.advance(err_task, 1)
            elif et == "flush":
                flush_state = evt.get("state", "unknown")
            elif et == "fatal":
                fatal = evt.get("error") or "unknown error"
                fatal_stage = evt.get("stage")
                fatal_hint = evt.get("hint")
            elif et == "close":
                flush_state = "closed"
                live.update(render())
                return

            live.update(render())



async def run_one_volume(args):
    # Import model loading and volume worker only when needed (requires torch)
    from .model_utils import load_model
    from .ld_volume_worker import LDVolumeWorker
    
    # Build PipelineConfig first to get config values for model loading
    cfg = PipelineConfig(
        s3_bucket="archive.tbrc.org",  # Default for S3 operations
        s3_region="us-east-1",
    )
    
    # Determine device for model loading
    device = None
    if cfg.use_gpu:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    
    # Load model with config options
    model = load_model(
        args.checkpoint,
        classes=1,
        device=device,
        precision=cfg.precision,
        compile_model=cfg.compile_model,
    )
    cfg.model = model
    
    # Set up debug configuration
    debug_images_set = None
    if args.debug_image:
        debug_images_set = set(args.debug_image)
    cfg.debug_mode = args.debug
    cfg.debug_images = debug_images_set
    
    # Build VolumeTask based on input mode
    volume_task: VolumeTask
    s3ctx: Optional[Any] = None  # S3Context imported lazily when needed
    
    if args.input_folder:
        # Local mode
        input_folder = os.path.abspath(args.input_folder)
        image_tasks = _get_local_image_tasks(input_folder)
        
        # Determine output folder
        if args.output_folder:
            output_base_uri = _normalize_uri(args.output_folder)
        else:
            # Default: output folder next to input folder
            output_base = str(Path(input_folder).parent / f"{Path(input_folder).name}_output")
            output_base_uri = _normalize_uri(output_base)
            # Create directory if local path
            if output_base_uri.startswith("file://"):
                path_part = output_base_uri[7:]
                if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
                    path_part = path_part[1:]
                os.makedirs(path_part, exist_ok=True)
        
        parquet_uri = _join_uri(output_base_uri, "results.parquet")
        jsonl_uri = _join_uri(output_base_uri, "errors.jsonl")
        
        # Debug folder (always local)
        if args.debug_folder:
            debug_folder = os.path.abspath(args.debug_folder)
        else:
            # Default: {output_folder}_debug/
            if output_base_uri.startswith("file://"):
                path_part = output_base_uri[7:]
                if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
                    path_part = path_part[1:]
                debug_folder = str(Path(path_part).parent / f"{Path(path_part).name}_debug")
            else:
                # For S3 output, use a local debug folder
                debug_folder = str(Path(input_folder).parent / f"{Path(input_folder).name}_debug")
        
        if cfg.debug_mode:
            os.makedirs(debug_folder, exist_ok=True)
        cfg.debug_folder = debug_folder if cfg.debug_mode else None
        
        volume_task = VolumeTask(
            io_mode="local",
            debug_folder_path=debug_folder,
            output_parquet_uri=parquet_uri,
            output_jsonl_uri=jsonl_uri,
            image_tasks=image_tasks,
        )
        
    elif args.w and args.i:
        # S3 mode
        w_id = args.w
        i_id = args.i
        
        image_tasks, i_version = get_image_list_and_version_s3(w_id, i_id)
        if image_tasks is None or i_version is None:
            raise ValueError(f"Failed to fetch image list for W{w_id} I{i_id}")
        
        # Determine output folder
        if args.output_folder:
            output_base_uri = _normalize_uri(args.output_folder).rstrip('/')
        else:
            output_base_uri = f"s3://tests-bec.bdrc.io/artefacts/line_detection_v1/{w_id}-{i_id}-{i_version}"
        
        parquet_filename = f"{w_id}-{i_id}-{i_version}.parquet"
        jsonl_filename = f"{w_id}-{i_id}-{i_version}.jsonl"
        
        parquet_uri = _join_uri(output_base_uri, parquet_filename)
        jsonl_uri = _join_uri(output_base_uri, jsonl_filename)
        
        # Debug folder (always local)
        if args.debug_folder:
            debug_folder = os.path.abspath(args.debug_folder)
        else:
            # Default: {output_folder}_debug/ (but output might be S3, so use home)
            if output_base_uri.startswith("file://"):
                path_part = output_base_uri[7:]
                if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
                    path_part = path_part[1:]
                debug_folder = str(Path(path_part).parent / f"{Path(path_part).name}_debug")
            else:
                debug_folder = str(Path.home() / f"debug_{w_id}_{i_id}")
        
        if cfg.debug_mode:
            os.makedirs(debug_folder, exist_ok=True)
        cfg.debug_folder = debug_folder if cfg.debug_mode else None
        
        volume_task = VolumeTask(
            io_mode="s3",
            debug_folder_path=debug_folder if cfg.debug_mode else str(Path.home() / f"debug_{w_id}_{i_id}"),  # Local debug folder
            output_parquet_uri=parquet_uri,
            output_jsonl_uri=jsonl_uri,
            image_tasks=image_tasks,
        )
        
        # Create S3Context for S3 mode (lazy import to avoid requiring aiobotocore)
        from .s3ctx import S3Context
        global_sem = asyncio.Semaphore(cfg.s3_max_inflight_global)
        s3ctx = S3Context(cfg, global_sem)
        
    else:
        raise ValueError("Either --input_folder OR both --w and --i must be provided")

    progress_events: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=10_000)
    hook = make_progress_hook(progress_events) if args.progress else None

    # Use async context manager for proper cleanup
    try:
        async with LDVolumeWorker(cfg, volume_task, s3ctx=s3ctx, progress=hook) as worker:
            total = len(volume_task.image_tasks)
            worker_run_t0: Optional[float] = None
            worker_run_s: Optional[float] = None

            if args.progress:
                ui = asyncio.create_task(
                    ui_loop(
                        events=progress_events,
                        worker=worker,
                        total=total,
                        show_queues=args.progress_queues,
                    )
                )
                try:
                    worker_run_t0 = time.perf_counter()
                    await worker.run()
                    worker_run_s = time.perf_counter() - worker_run_t0
                finally:
                    if worker_run_s is None and worker_run_t0 is not None:
                        worker_run_s = time.perf_counter() - worker_run_t0
                    # If pipeline dies early, unblock UI.
                    try:
                        progress_events.put_nowait({"type": "close"})
                    except asyncio.QueueFull:
                        pass
                    await ui
                    # Logging after Live has exited to avoid mangling the UI.
                    if worker_run_s is not None:
                        ips = (float(total) / worker_run_s) if (total and worker_run_s > 0) else 0.0
                        console.log(f"Volume worker runtime: {worker_run_s:.3f}s (images={total}, {ips:.2f} img/s)")
            else:
                worker_run_t0 = time.perf_counter()
                await worker.run()
                worker_run_s = time.perf_counter() - worker_run_t0
                ips = (float(total) / worker_run_s) if (total and worker_run_s > 0) else 0.0
                console.log(f"Volume worker runtime: {worker_run_s:.3f}s (images={total}, {ips:.2f} img/s)")
    finally:
        # Close S3 client if used
        if s3ctx is not None:
            await s3ctx.close()


def main():
    p = argparse.ArgumentParser(description="Line detection pipeline for OCR inference")
    
    # Input mode
    p.add_argument(
        "--input-folder",
        type=str,
        help="Local folder containing image files (creates local VolumeTask)"
    )
    s3_group = p.add_argument_group("S3 mode (requires both --w and --i)")
    s3_group.add_argument(
        "--w",
        type=str,
        help="Work ID (e.g., W22084) for S3 mode"
    )
    s3_group.add_argument(
        "--i",
        type=str,
        help="Image group ID (e.g., I0886) for S3 mode"
    )
    
    # Output
    p.add_argument(
        "--output-folder",
        type=str,
        help="Output folder path or URI (file:// or s3://). Defaults set based on input mode."
    )
    
    # Model
    p.add_argument(
        "-c", "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    
    # UI
    p.add_argument("--progress", action="store_true", help="Show progress UI on stderr.")
    p.add_argument(
        "--progress-queues",
        action="store_true",
        help="Also show queue fullness (updates ~1s). Implies --progress.",
    )
    
    # Debug
    p.add_argument("--debug", action="store_true", help="Enable debug mode to output intermediate data.")
    p.add_argument(
        "--debug-folder",
        type=str,
        help="Debug output folder (defaults to {output_folder}_debug/). Implies --debug.",
    )
    p.add_argument(
        "--debug-image",
        action="append",
        help="Enable debug output only for specified image filename(s). Can be specified multiple times. Implies --debug.",
    )
    
    # Logging
    p.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set logging level (default: WARNING). Use INFO to see timing logs.",
    )
    
    args = p.parse_args()
    
    # Validate input mode arguments
    if args.input_folder:
        if args.w or args.i:
            p.error("--input-folder cannot be used with --w or --i")
    else:
        if not (args.w and args.i):
            p.error("Either --input-folder OR both --w and --i must be provided")
    
    if args.progress_queues:
        args.progress = True
    
    # Handle debug flags
    if args.debug_folder or args.debug_image:
        args.debug = True
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run_one_volume(args))


if __name__ == "__main__":
    main()
