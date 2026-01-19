"""
BEC Workers CLI - Manage and inspect workers.

Usage:
  bec workers list [--active] [--limit N]
  bec workers show <worker-id>
  bec workers stats [--active]
"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

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


def resolve_worker_identifier(conn, worker_identifier: str) -> int:
    """
    Resolve worker identifier (ID or name) to worker_id.
    
    Args:
        conn: Database connection
        worker_identifier: Worker ID (numeric string) or worker name
        
    Returns:
        Worker ID as integer
        
    Raises:
        ValueError: If worker not found
    """
    # Try to parse as integer (worker ID)
    try:
        worker_id = int(worker_identifier)
        # Verify it exists
        with conn.cursor() as cur:
            cur.execute("SELECT worker_id FROM workers WHERE worker_id = %s", (worker_id,))
            if cur.fetchone():
                return worker_id
            else:
                raise ValueError(f"Worker ID {worker_id} not found")
    except ValueError:
        # Not an integer, treat as worker name
        with conn.cursor() as cur:
            cur.execute("SELECT worker_id FROM workers WHERE worker_name = %s", (worker_identifier,))
            row = cur.fetchone()
            if row:
                return row['worker_id']
            else:
                raise ValueError(f"Worker '{worker_identifier}' not found")


@click.group()
def workers():
    """Manage and inspect workers."""
    pass


@workers.command("list")
@click.option("--active", is_flag=True, help="Show only active workers (heartbeat within last 10 minutes)")
@click.option("--limit", "-n", type=int, default=50, help="Number of workers to show (default: 50)")
def list_workers(active: bool, limit: int):
    """List workers with their last activity and recent jobs.
    
    Workers are ordered by most recent activity (last_heartbeat_at).
    Active workers are those with heartbeat within the last 10 minutes.
    """
    conn = get_db_connection()
    try:
        # Build query
        conditions = []
        if active:
            conditions.append("w.last_heartbeat_at >= NOW() - INTERVAL '10 minutes'")
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            WITH latest_tasks AS (
                SELECT DISTINCT ON (worker_id)
                    worker_id,
                    job_id,
                    status,
                    started_at,
                    done_at,
                    volume_id
                FROM task_executions
                WHERE worker_id IS NOT NULL
                ORDER BY worker_id, id DESC
            )
            SELECT 
                w.worker_id,
                w.worker_name,
                w.hostname,
                w.started_at as worker_started_at,
                w.last_heartbeat_at,
                w.stopped_at,
                EXTRACT(EPOCH FROM (NOW() - w.last_heartbeat_at)) as seconds_since_heartbeat,
                j.name as last_job_name,
                lt.status as last_task_status,
                lt.started_at as last_task_started_at,
                lt.done_at as last_task_done_at,
                v.bdrc_w_id,
                v.bdrc_i_id,
                (SELECT COUNT(*) FROM task_executions te WHERE te.worker_id = w.worker_id) as total_tasks,
                (SELECT COUNT(*) FROM task_executions te WHERE te.worker_id = w.worker_id AND te.status = 'done') as completed_tasks,
                (SELECT COUNT(*) FROM task_executions te WHERE te.worker_id = w.worker_id AND te.status IN ('retryable_failed', 'terminal_failed')) as failed_tasks
            FROM workers w
            LEFT JOIN latest_tasks lt ON w.worker_id = lt.worker_id
            LEFT JOIN jobs j ON lt.job_id = j.id
            LEFT JOIN volumes v ON lt.volume_id = v.id
            {where_clause}
            ORDER BY w.last_heartbeat_at DESC
            LIMIT %s
        """
        
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
        
        if not rows:
            console.print("[yellow]No workers found[/yellow]")
            return
        
        # Display table
        title = f"Workers (showing {len(rows)})"
        if active:
            title += " - Active Only"
        
        table = Table(title=title)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("Last Activity", style="dim")
        table.add_column("Last Job", style="blue")
        table.add_column("Last Volume", style="yellow")
        table.add_column("Last Status")
        table.add_column("Tasks", justify="right")
        table.add_column("Done", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        
        for row in rows:
            # Determine worker status
            seconds_since = row["seconds_since_heartbeat"]
            if row["stopped_at"]:
                status = "[red]stopped[/red]"
            elif seconds_since < 60:
                status = "[green]active[/green]"
            elif seconds_since < 600:  # 10 minutes
                status = "[yellow]idle[/yellow]"
            else:
                status = "[dim]inactive[/dim]"
            
            # Format last activity time
            if seconds_since < 60:
                activity = f"{int(seconds_since)}s ago"
            elif seconds_since < 3600:
                activity = f"{int(seconds_since / 60)}m ago"
            elif seconds_since < 86400:
                activity = f"{int(seconds_since / 3600)}h ago"
            else:
                activity = f"{int(seconds_since / 86400)}d ago"
            
            # Worker name (never truncate - EC2 instance IDs must be visible)
            worker_name = row["worker_name"] or "-"
            
            # Last job and volume
            last_job = row["last_job_name"] or "-"
            if row["bdrc_w_id"] and row["bdrc_i_id"]:
                last_volume = f"{row['bdrc_w_id']}/{row['bdrc_i_id']}"
                if len(last_volume) > 25:
                    last_volume = last_volume[:22] + "..."
            else:
                last_volume = "-"
            
            # Last task status
            last_status = row["last_task_status"] or "-"
            if last_status != "-":
                status_style = {
                    "running": "yellow",
                    "done": "green",
                    "retryable_failed": "orange1",
                    "terminal_failed": "red",
                }.get(last_status, "white")
                last_status = f"[{status_style}]{last_status}[/{status_style}]"
            
            table.add_row(
                str(row["worker_id"]),
                worker_name,
                status,
                activity,
                last_job,
                last_volume,
                last_status,
                str(row["total_tasks"]),
                str(row["completed_tasks"]),
                str(row["failed_tasks"]),
            )
        
        console.print(table)
        console.print()
        console.print("[dim]Use 'bec workers show <worker-id>' for detailed worker information[/dim]")
        
    finally:
        conn.close()


@workers.command("show")
@click.argument("worker_identifier")
@click.option("--limit", "-n", type=int, default=10, help="Number of recent tasks to show (default: 10)")
def show_worker(worker_identifier: str, limit: int):
    """Show detailed information about a specific worker.
    
    Displays worker metadata, activity timeline, and recent task executions.
    Worker can be specified by ID or name (EC2 instance ID).
    """
    conn = get_db_connection()
    try:
        # Resolve worker identifier to worker_id
        try:
            worker_id = resolve_worker_identifier(conn, worker_identifier)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        
        # Get worker info
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    worker_id,
                    worker_name,
                    hostname,
                    tags,
                    started_at,
                    last_heartbeat_at,
                    stopped_at,
                    EXTRACT(EPOCH FROM (NOW() - last_heartbeat_at)) as seconds_since_heartbeat
                FROM workers
                WHERE worker_id = %s
                """,
                (worker_id,)
            )
            worker = cur.fetchone()
        
        if not worker:
            console.print(f"[red]Error:[/red] Worker {worker_id} not found")
            sys.exit(1)
        
        # Display worker info
        console.print(f"\n[bold]Worker {worker_id}[/bold]")
        console.print(f"Name: [green]{worker['worker_name'] or 'N/A'}[/green]")
        console.print(f"Hostname: {worker['hostname'] or 'N/A'}")
        
        if worker["tags"]:
            console.print(f"Tags: {worker['tags']}")
        
        console.print(f"Started: {worker['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Last Heartbeat: {worker['last_heartbeat_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        seconds_since = worker["seconds_since_heartbeat"]
        if seconds_since < 60:
            activity = f"{int(seconds_since)} seconds ago"
        elif seconds_since < 3600:
            activity = f"{int(seconds_since / 60)} minutes ago"
        elif seconds_since < 86400:
            activity = f"{int(seconds_since / 3600)} hours ago"
        else:
            activity = f"{int(seconds_since / 86400)} days ago"
        console.print(f"Activity: {activity}")
        
        if worker["stopped_at"]:
            console.print(f"Stopped: [red]{worker['stopped_at'].strftime('%Y-%m-%d %H:%M:%S')}[/red]")
        
        # Get task statistics
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_tasks,
                    COUNT(*) FILTER (WHERE status = 'running') as running,
                    COUNT(*) FILTER (WHERE status = 'done') as done,
                    COUNT(*) FILTER (WHERE status = 'retryable_failed') as retryable_failed,
                    COUNT(*) FILTER (WHERE status = 'terminal_failed') as terminal_failed,
                    SUM(total_images) as total_images,
                    SUM(nb_errors) as total_errors,
                    ROUND(AVG(total_duration_ms)::numeric, 2) as avg_duration_ms,
                    ROUND(AVG(avg_duration_per_page_ms)::numeric, 2) as avg_per_page_ms
                FROM task_executions
                WHERE worker_id = %s
                """,
                (worker_id,)
            )
            stats = cur.fetchone()
        
        console.print(f"\n[bold]Task Statistics:[/bold]")
        console.print(f"  Total Tasks: {stats['total_tasks']}")
        console.print(f"  Completed: [green]{stats['done']}[/green]")
        if stats['retryable_failed'] > 0:
            console.print(f"  Retryable Failed: [orange1]{stats['retryable_failed']}[/orange1]")
        if stats['terminal_failed'] > 0:
            console.print(f"  Terminal Failed: [red]{stats['terminal_failed']}[/red]")
        if stats['running'] > 0:
            console.print(f"  Running: [yellow]{stats['running']}[/yellow]")
        
        if stats['total_images']:
            console.print(f"  Total Images: {stats['total_images']}")
        if stats['total_errors']:
            console.print(f"  Total Errors: {stats['total_errors']}")
        if stats['avg_duration_ms']:
            console.print(f"  Avg Duration per Task: {stats['avg_duration_ms']:.2f} ms")
        if stats['avg_per_page_ms']:
            console.print(f"  Avg Duration per Page: {stats['avg_per_page_ms']:.2f} ms")
        
        # Get recent tasks
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    te.id,
                    j.name as job_name,
                    v.bdrc_w_id,
                    v.bdrc_i_id,
                    te.status,
                    te.started_at,
                    te.done_at,
                    te.total_images,
                    te.nb_errors,
                    te.total_duration_ms
                FROM task_executions te
                JOIN jobs j ON te.job_id = j.id
                JOIN volumes v ON te.volume_id = v.id
                WHERE te.worker_id = %s
                ORDER BY te.id DESC
                LIMIT %s
                """,
                (worker_id, limit)
            )
            recent_tasks = cur.fetchall()
        
        if recent_tasks:
            console.print(f"\n[bold]Recent Task Executions (last {len(recent_tasks)}):[/bold]")
            
            table = Table()
            table.add_column("Task ID", style="cyan")
            table.add_column("Job", style="blue")
            table.add_column("Volume", style="green")
            table.add_column("Status")
            table.add_column("Images", justify="right")
            table.add_column("Errors", justify="right")
            table.add_column("Duration (s)", justify="right")
            table.add_column("Completed", style="dim")
            
            for task in recent_tasks:
                status_style = {
                    "running": "yellow",
                    "done": "green",
                    "retryable_failed": "orange1",
                    "terminal_failed": "red",
                }.get(task["status"], "white")
                
                duration_s = f"{task['total_duration_ms'] / 1000:.1f}" if task['total_duration_ms'] else "-"
                completed_at = task['done_at'].strftime("%Y-%m-%d %H:%M") if task['done_at'] else "-"
                
                table.add_row(
                    str(task["id"]),
                    task["job_name"],
                    f"{task['bdrc_w_id']}/{task['bdrc_i_id']}",
                    f"[{status_style}]{task['status']}[/{status_style}]",
                    str(task["total_images"] or "-"),
                    str(task["nb_errors"] or "-"),
                    duration_s,
                    completed_at,
                )
            
            console.print(table)
        
        console.print()
        
    finally:
        conn.close()


@workers.command("stats")
@click.option("--active", is_flag=True, help="Show stats for active workers only (heartbeat within last 10 minutes)")
def worker_stats(active: bool):
    """Show aggregate statistics across all workers.
    
    Displays task completion rates, error rates, and performance metrics.
    """
    conn = get_db_connection()
    try:
        # Build worker filter
        worker_filter = ""
        if active:
            worker_filter = """
                AND te.worker_id IN (
                    SELECT worker_id FROM workers 
                    WHERE last_heartbeat_at >= NOW() - INTERVAL '10 minutes'
                )
            """
        
        # Get overall stats
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    COUNT(DISTINCT te.worker_id) as total_workers,
                    COUNT(*) as total_tasks,
                    COUNT(*) FILTER (WHERE te.status = 'done') as done_tasks,
                    COUNT(*) FILTER (WHERE te.status = 'retryable_failed') as retryable_failed,
                    COUNT(*) FILTER (WHERE te.status = 'terminal_failed') as terminal_failed,
                    SUM(te.total_images) as total_images,
                    SUM(te.nb_errors) as total_errors,
                    ROUND(AVG(te.total_duration_ms)::numeric, 2) as avg_duration_ms,
                    ROUND(AVG(te.avg_duration_per_page_ms)::numeric, 2) as avg_per_page_ms
                FROM task_executions te
                WHERE te.worker_id IS NOT NULL
                {worker_filter}
                """
            )
            overall = cur.fetchone()
        
        # Get per-worker stats
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    w.worker_id,
                    w.worker_name,
                    COUNT(*) as total_tasks,
                    COUNT(*) FILTER (WHERE te.status = 'done') as done_tasks,
                    COUNT(*) FILTER (WHERE te.status IN ('retryable_failed', 'terminal_failed')) as failed_tasks,
                    SUM(te.total_images) as total_images,
                    ROUND(AVG(te.total_duration_ms)::numeric, 2) as avg_duration_ms
                FROM workers w
                JOIN task_executions te ON w.worker_id = te.worker_id
                {"WHERE w.last_heartbeat_at >= NOW() - INTERVAL '10 minutes'" if active else ""}
                GROUP BY w.worker_id, w.worker_name
                ORDER BY total_tasks DESC
                LIMIT 20
                """
            )
            per_worker = cur.fetchall()
        
        # Display overall stats
        title = "Worker Statistics"
        if active:
            title += " (Active Workers Only)"
        
        console.print(f"\n[bold]{title}[/bold]\n")
        console.print(f"Total Workers: {overall['total_workers']}")
        console.print(f"Total Tasks: {overall['total_tasks']}")
        console.print(f"Completed: [green]{overall['done_tasks']}[/green] ({overall['done_tasks'] * 100 // overall['total_tasks'] if overall['total_tasks'] > 0 else 0}%)")
        
        if overall['retryable_failed'] > 0:
            console.print(f"Retryable Failed: [orange1]{overall['retryable_failed']}[/orange1] ({overall['retryable_failed'] * 100 // overall['total_tasks']}%)")
        if overall['terminal_failed'] > 0:
            console.print(f"Terminal Failed: [red]{overall['terminal_failed']}[/red] ({overall['terminal_failed'] * 100 // overall['total_tasks']}%)")
        
        if overall['total_images']:
            console.print(f"Total Images: {overall['total_images']}")
        if overall['total_errors']:
            error_rate = overall['total_errors'] * 100.0 / overall['total_images'] if overall['total_images'] > 0 else 0
            console.print(f"Total Errors: {overall['total_errors']} ({error_rate:.2f}%)")
        if overall['avg_duration_ms']:
            console.print(f"Avg Duration per Task: {overall['avg_duration_ms']:.2f} ms")
        if overall['avg_per_page_ms']:
            console.print(f"Avg Duration per Page: {overall['avg_per_page_ms']:.2f} ms")
        
        # Display per-worker stats
        if per_worker:
            console.print(f"\n[bold]Top Workers by Task Count:[/bold]")
            
            table = Table()
            table.add_column("Worker ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Total Tasks", justify="right")
            table.add_column("Done", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Images", justify="right")
            table.add_column("Avg Duration (ms)", justify="right")
            
            for row in per_worker:
                worker_name = row["worker_name"] or "-"
                
                table.add_row(
                    str(row["worker_id"]),
                    worker_name,
                    str(row["total_tasks"]),
                    str(row["done_tasks"]),
                    str(row["failed_tasks"]),
                    str(row["total_images"] or 0),
                    str(row["avg_duration_ms"] or "-"),
                )
            
            console.print(table)
        
        console.print()
        
    finally:
        conn.close()
