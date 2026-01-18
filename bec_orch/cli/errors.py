"""
BEC Errors CLI - Inspect task execution errors.

Usage:
  bec errors list [--job JOB] [--type TYPE] [--limit N]
  bec errors show W12345/I0001 [--job JOB]
  bec errors recent [--job JOB] [--limit N]
"""

import json
import os
import sys
from typing import Optional

import boto3
import click
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()


def get_db_connection():
    """Get database connection from environment."""
    from bec_orch.io.db import DBClient, build_dsn_from_env

    try:
        dsn = build_dsn_from_env()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    db = DBClient(dsn)
    return db.connect()


def get_s3_client():
    """Get S3 client with region from environment."""
    region = os.environ.get("BEC_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def get_dest_bucket() -> str:
    """Get destination S3 bucket from environment."""
    bucket = os.environ.get("BEC_DEST_S3_BUCKET")
    if not bucket:
        console.print(
            "[red]Error:[/red] BEC_DEST_S3_BUCKET environment variable not set.\n"
            "Set it to the S3 bucket where job artifacts are stored."
        )
        sys.exit(1)
    return bucket


def compute_errors_jsonl_key(job_name: str, w_id: str, i_id: str, etag: str) -> str:
    """
    Compute the S3 key for errors.jsonl file.

    Format: {job_name}/{w_id}/{i_id}/{version}/{w_id}-{i_id}-{version}-errors.jsonl
    Where version is first 6 chars of etag (without quotes).
    """
    version = etag.replace('"', "").split("-")[0][:6]
    prefix = f"{job_name}/{w_id}/{i_id}/{version}"
    basename = f"{w_id}-{i_id}-{version}"
    return f"{prefix}/{basename}-errors.jsonl"


def fetch_errors_jsonl(s3, bucket: str, key: str) -> list[dict]:
    """Fetch and parse errors.jsonl from S3."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        errors = []
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    errors.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return errors
    except s3.exceptions.NoSuchKey:
        return []
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return []
        raise


@click.group()
def errors():
    """Inspect task execution errors."""
    pass


@errors.command("list")
@click.option("--job", "-j", help="Filter by job name")
@click.option(
    "--type",
    "-t",
    "error_type",
    type=click.Choice(["all", "task-failed", "image-errors"]),
    default="all",
    help="Type of errors: all (default), task-failed, image-errors",
)
@click.option("--limit", "-n", type=int, default=20, help="Number of results (default: 20)")
def list_errors(job: Optional[str], error_type: str, limit: int):
    """List recent task executions with errors.

    Shows tasks that either failed (retryable_failed/terminal_failed) or
    completed with image-level errors (nb_errors > 0).
    """
    conn = get_db_connection()
    try:
        # Build query based on error type filter
        conditions = []
        params = []

        if error_type == "task-failed":
            conditions.append("te.status IN ('retryable_failed', 'terminal_failed')")
        elif error_type == "image-errors":
            conditions.append("te.status = 'done' AND te.nb_errors > 0")
        else:  # all
            conditions.append(
                "(te.status IN ('retryable_failed', 'terminal_failed') OR "
                "(te.status = 'done' AND te.nb_errors > 0))"
            )

        if job:
            conditions.append("j.name = %s")
            params.append(job)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT 
                te.id,
                v.bdrc_w_id,
                v.bdrc_i_id,
                j.name as job_name,
                te.status,
                te.nb_errors,
                te.total_images,
                te.done_at,
                te.started_at,
                encode(te.s3_etag, 'hex') as etag_hex
            FROM task_executions te
            JOIN volumes v ON te.volume_id = v.id
            JOIN jobs j ON te.job_id = j.id
            WHERE {where_clause}
            ORDER BY COALESCE(te.done_at, te.started_at) DESC
            LIMIT %s
        """
        params.append(limit)

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            console.print("[yellow]No errors found[/yellow]")
            return

        # Display table
        table = Table(title=f"Task Executions with Errors (showing {len(rows)})")
        table.add_column("ID", style="cyan")
        table.add_column("Volume", style="green")
        table.add_column("Job", style="blue")
        table.add_column("Status")
        table.add_column("Errors", justify="right")
        table.add_column("Images", justify="right")
        table.add_column("Time", style="dim")

        for row in rows:
            status = row["status"]
            status_style = {
                "running": "yellow",
                "done": "green",
                "retryable_failed": "orange1",
                "terminal_failed": "red",
            }.get(status, "white")

            volume = f"{row['bdrc_w_id']}/{row['bdrc_i_id']}"
            time_str = (
                row["done_at"].strftime("%Y-%m-%d %H:%M")
                if row["done_at"]
                else row["started_at"].strftime("%Y-%m-%d %H:%M")
                if row["started_at"]
                else "-"
            )

            table.add_row(
                str(row["id"]),
                volume,
                row["job_name"],
                f"[{status_style}]{status}[/{status_style}]",
                str(row["nb_errors"] or 0),
                str(row["total_images"] or "-"),
                time_str,
            )

        console.print(table)
        console.print()
        console.print("[dim]Use 'bec errors show W.../I...' to see detailed errors for a volume[/dim]")

    finally:
        conn.close()


@errors.command("show")
@click.argument("volume", required=False)
@click.option("--w", "w_id", help="Work ID (alternative to positional volume)")
@click.option("--i", "i_id", help="Image group ID (alternative to positional volume)")
@click.option("--job", "-j", required=True, help="Job name (required)")
@click.option("--full", is_flag=True, help="Show full error details including traceback")
def show_errors(volume: Optional[str], w_id: Optional[str], i_id: Optional[str], job: str, full: bool):
    """Show detailed errors for a specific volume.

    Specify volume as W12345/I0001 or use --w and --i flags.
    """
    # Parse volume identifier
    if volume:
        if "/" not in volume:
            console.print("[red]Error:[/red] Volume must be in format W.../I... (e.g., W22084/I0886)")
            sys.exit(1)
        parts = volume.split("/")
        w_id = parts[0]
        i_id = parts[1]
    elif not w_id or not i_id:
        console.print("[red]Error:[/red] Specify volume as W.../I... or use --w and --i flags")
        sys.exit(1)

    conn = get_db_connection()
    try:
        # Get task execution with etag
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    te.id,
                    te.status,
                    te.nb_errors,
                    te.total_images,
                    encode(te.s3_etag, 'hex') as etag_hex,
                    j.name as job_name
                FROM task_executions te
                JOIN volumes v ON te.volume_id = v.id
                JOIN jobs j ON te.job_id = j.id
                WHERE v.bdrc_w_id = %s AND v.bdrc_i_id = %s AND j.name = %s
                ORDER BY te.id DESC
                LIMIT 1
                """,
                (w_id, i_id, job),
            )
            row = cur.fetchone()

        if not row:
            console.print(f"[yellow]No task execution found for {w_id}/{i_id} in job '{job}'[/yellow]")
            return

        console.print(f"\n[bold]Errors for {w_id}/{i_id}[/bold] (job={job})")
        console.print(f"Task ID: {row['id']}, Status: {row['status']}, Errors: {row['nb_errors'] or 0}/{row['total_images'] or '?'} images")
        console.print()

        # Fetch errors.jsonl from S3
        s3 = get_s3_client()
        bucket = get_dest_bucket()
        key = compute_errors_jsonl_key(job, w_id, i_id, row["etag_hex"])

        console.print(f"[dim]Fetching: s3://{bucket}/{key}[/dim]")

        errors_list = fetch_errors_jsonl(s3, bucket, key)

        if not errors_list:
            console.print("[yellow]No errors.jsonl file found or file is empty[/yellow]")
            console.print("[dim]This may indicate a task-level failure (no image processing started)[/dim]")
            return

        console.print(f"\n[bold]Found {len(errors_list)} errors:[/bold]\n")

        for i, err in enumerate(errors_list, 1):
            img = err.get("img_filename")
            stage = err.get("stage", "?")
            err_type = err.get("error_type", "?")
            message = err.get("message", "")

            # Handle volume-level errors (no specific image or empty string)
            if img and img.strip():
                console.print(f"[cyan]{i}.[/cyan] [green]{img}[/green]")
            else:
                console.print(f"[cyan]{i}.[/cyan] [bold magenta]⚠ VOLUME-LEVEL ERROR[/bold magenta]")
            
            console.print(f"   Stage: [yellow]{stage}[/yellow], Type: [red]{err_type}[/red]")
            
            # For VolumeTimeout, parse diagnostic from message or show truncated message
            if err_type == "VolumeTimeout" and "Diagnostic:" in message:
                # Extract and display diagnostic state from message
                try:
                    import json
                    diag_start = message.index("Diagnostic:") + len("Diagnostic:")
                    diag_json = message[diag_start:].strip()
                    diag = json.loads(diag_json)
                    console.print("   [dim]Pipeline timed out. Queue state:[/dim]")
                    if isinstance(diag, dict) and "queues" in diag:
                        for q_name, q_state in diag["queues"].items():
                            console.print(f"     [dim]{q_name}: {q_state}[/dim]")
                except (ValueError, json.JSONDecodeError):
                    console.print(f"   Message: {message[:200]}{'...' if len(message) > 200 else ''}")
            else:
                console.print(f"   Message: {message[:200]}{'...' if len(message) > 200 else ''}")

            if full and err.get("traceback"):
                console.print("   [dim]Traceback:[/dim]")
                console.print(Syntax(err["traceback"], "python", theme="monokai", line_numbers=False))

            console.print()

    finally:
        conn.close()


@errors.command("recent")
@click.option("--job", "-j", help="Filter by job name")
@click.option("--limit", "-n", type=int, default=10, help="Number of errors to show (default: 10)")
@click.option("--full", is_flag=True, help="Show full error details including traceback")
def recent_errors(job: Optional[str], limit: int, full: bool):
    """Show the most recent individual errors across all volumes.

    Fetches errors.jsonl files from recent task executions and displays
    the most recent individual error entries.
    """
    conn = get_db_connection()
    try:
        # Find recent task executions with image-level errors
        conditions = ["te.status = 'done'", "te.nb_errors > 0"]
        params = []

        if job:
            conditions.append("j.name = %s")
            params.append(job)

        where_clause = " AND ".join(conditions)

        # Fetch more than we need since we'll be aggregating individual errors
        query = f"""
            SELECT 
                te.id,
                v.bdrc_w_id,
                v.bdrc_i_id,
                j.name as job_name,
                te.nb_errors,
                te.done_at,
                encode(te.s3_etag, 'hex') as etag_hex
            FROM task_executions te
            JOIN volumes v ON te.volume_id = v.id
            JOIN jobs j ON te.job_id = j.id
            WHERE {where_clause}
            ORDER BY te.done_at DESC
            LIMIT 50
        """

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            console.print("[yellow]No task executions with image errors found[/yellow]")
            return

        # Fetch errors from S3 for each task execution
        s3 = get_s3_client()
        bucket = get_dest_bucket()

        all_errors = []
        console.print("[dim]Fetching error files...[/dim]")

        for row in rows:
            key = compute_errors_jsonl_key(
                row["job_name"], row["bdrc_w_id"], row["bdrc_i_id"], row["etag_hex"]
            )
            try:
                errors_list = fetch_errors_jsonl(s3, bucket, key)
                for err in errors_list:
                    err["_volume"] = f"{row['bdrc_w_id']}/{row['bdrc_i_id']}"
                    err["_job"] = row["job_name"]
                    err["_done_at"] = row["done_at"]
                    all_errors.append(err)
            except Exception as e:
                console.print(f"[dim]Warning: Could not fetch {key}: {e}[/dim]")

            # Stop early if we have enough
            if len(all_errors) >= limit * 2:
                break

        if not all_errors:
            console.print("[yellow]No error details found in S3[/yellow]")
            return

        # Sort by time (most recent first) and take top N
        all_errors.sort(key=lambda x: x.get("_done_at") or "", reverse=True)
        recent = all_errors[:limit]

        console.print(f"\n[bold]Most Recent {len(recent)} Errors:[/bold]\n")

        for i, err in enumerate(recent, 1):
            volume = err.get("_volume", "?")
            img = err.get("img_filename", "unknown")
            stage = err.get("stage", "?")
            err_type = err.get("error_type", "?")
            message = err.get("message", "")

            console.print(f"[cyan]{i}.[/cyan] [blue]{volume}[/blue] / [green]{img}[/green]")
            console.print(f"   Stage: [yellow]{stage}[/yellow], Type: [red]{err_type}[/red]")
            console.print(f"   Message: {message[:200]}{'...' if len(message) > 200 else ''}")

            if full and err.get("traceback"):
                console.print("   [dim]Traceback:[/dim]")
                console.print(Syntax(err["traceback"], "python", theme="monokai", line_numbers=False))

            console.print()

    finally:
        conn.close()
