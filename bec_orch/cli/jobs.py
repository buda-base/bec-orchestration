"""
BEC Jobs CLI - Manage jobs.

Usage:
  bec jobs create --name ldv1 --config config.json
  bec jobs list
  bec jobs show <job-id>
  bec jobs update <job-id> --config config.json
  bec jobs delete <job-id>
"""

import json
import os
import sys
from pathlib import Path

import boto3
import click
from rich.console import Console
from rich.table import Table

console = Console()


def get_db_connection():
    """Get database connection from environment."""
    from bec_orch.io.db import DBClient
    
    sql_host = os.environ.get('BEC_SQL_HOST')
    sql_port = os.environ.get('BEC_SQL_PORT', '5432')
    sql_user = os.environ.get('BEC_SQL_USER')
    sql_password = os.environ.get('BEC_SQL_PASSWORD')
    sql_database = os.environ.get('BEC_SQL_DATABASE', 'pipeline_v1')
    
    if not all([sql_host, sql_user, sql_password]):
        console.print("[red]Error:[/red] Missing required SQL environment variables")
        console.print("Required: BEC_SQL_HOST, BEC_SQL_USER, BEC_SQL_PASSWORD")
        sys.exit(1)
    
    dsn = f"postgresql://{sql_user}:{sql_password}@{sql_host}:{sql_port}/{sql_database}"
    db = DBClient(dsn)
    return db.connect()


@click.group()
def jobs():
    """Manage jobs (create, show, update, delete)."""
    pass


@jobs.command()
@click.option('--name', required=True, help='Job name (e.g., ldv1, ocr) - must be unique')
@click.option('--config', type=click.Path(exists=True), help='Path to config file (JSON)')
@click.option('--config-text', help='Config text directly (JSON string)')
@click.option('--queue-url', help='SQS queue URL (optional: auto-creates bec_{name}_tasks if not provided)')
@click.option('--dlq-url', help='SQS DLQ URL (optional: auto-creates bec_{name}_dlq if not provided)')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def create(name, config, config_text, queue_url, dlq_url, region):
    """Create a new job with automatic SQS queue creation.
    
    By default, creates queues with naming convention:
      Queue:  bec_{job_name}_tasks
      DLQ:    bec_{job_name}_dlq
    
    You can also provide existing queue URLs with --queue-url and --dlq-url.
    """
    from bec_orch.orch.job_admin import create_job
    
    # Get config text from file or argument
    if config and config_text:
        console.print("[red]Error:[/red] Cannot specify both --config and --config-text")
        sys.exit(1)
    
    if config:
        config_text = Path(config).read_text()
        # Validate JSON
        try:
            json.loads(config_text)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in config file: {e}")
            sys.exit(1)
    elif config_text:
        # Validate JSON
        try:
            json.loads(config_text)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in config: {e}")
            sys.exit(1)
    else:
        config_text = "{}"  # Empty config
    
    # Get AWS region
    region = region or os.environ.get('BEC_REGION', 'us-east-1')
    
    # Create queues if not provided
    if not queue_url:
        console.print(f"[cyan]Creating SQS queues for job '{name}'...[/cyan]")
        queue_url, dlq_url = _create_job_queues(name, region)
        console.print(f"[green]✓[/green] Created queue: {queue_url}")
        console.print(f"[green]✓[/green] Created DLQ: {dlq_url}")
    elif not dlq_url:
        # Queue provided but not DLQ - create DLQ
        console.print(f"[cyan]Creating DLQ for job '{name}'...[/cyan]")
        dlq_name = f"bec_{name}_dlq"
        dlq_url = _create_queue(dlq_name, region, is_dlq=True)
        console.print(f"[green]✓[/green] Created DLQ: {dlq_url}")
    
    conn = get_db_connection()
    try:
        job_id = create_job(conn, name, queue_url, config_text, dlq_url)
        console.print(f"[green]✓[/green] Created job: id={job_id}, name={name}")
        console.print(f"Queue: {queue_url}")
        if dlq_url:
            console.print(f"DLQ: {dlq_url}")
    finally:
        conn.close()


def _create_job_queues(job_name: str, region: str) -> tuple[str, str]:
    """
    Create SQS queues for a job following convention.
    
    Creates:
      - Main queue: bec_{job_name}_tasks
      - DLQ: bec_{job_name}_dlq
    
    Returns:
        (queue_url, dlq_url)
    """
    queue_name = f"bec_{job_name}_tasks"
    dlq_name = f"bec_{job_name}_dlq"
    
    # Create DLQ first
    dlq_url = _create_queue(dlq_name, region, is_dlq=True)
    
    # Get DLQ ARN
    sqs = boto3.client('sqs', region_name=region)
    dlq_attrs = sqs.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=['QueueArn']
    )
    dlq_arn = dlq_attrs['Attributes']['QueueArn']
    
    # Create main queue with DLQ redrive policy
    queue_url = _create_queue(
        queue_name,
        region,
        is_dlq=False,
        dlq_arn=dlq_arn,
        max_receive_count=3
    )
    
    return queue_url, dlq_url


def _create_queue(
    queue_name: str,
    region: str,
    is_dlq: bool = False,
    dlq_arn: str = None,
    max_receive_count: int = 3
) -> str:
    """
    Create an SQS queue.
    
    Args:
        queue_name: Queue name
        region: AWS region
        is_dlq: Whether this is a dead-letter queue
        dlq_arn: DLQ ARN (for main queue)
        max_receive_count: Max receive count before sending to DLQ
        
    Returns:
        Queue URL
    """
    sqs = boto3.client('sqs', region_name=region)
    
    attributes = {
        'MessageRetentionPeriod': '345600',  # 4 days
        'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
    }
    
    # Add visibility timeout
    if not is_dlq:
        attributes['VisibilityTimeout'] = '300'  # 5 minutes
    
    # Add redrive policy for main queue
    if dlq_arn:
        attributes['RedrivePolicy'] = json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': str(max_receive_count)
        })
    
    try:
        response = sqs.create_queue(
            QueueName=queue_name,
            Attributes=attributes
        )
        return response['QueueUrl']
    except sqs.exceptions.QueueNameExists:
        # Queue already exists, get its URL
        response = sqs.get_queue_url(QueueName=queue_name)
        return response['QueueUrl']


@jobs.command()
def list():
    """List all jobs."""
    from bec_orch.orch.job_admin import list_jobs
    
    conn = get_db_connection()
    try:
        job_list = list_jobs(conn)
        
        if not job_list:
            console.print("[yellow]No jobs found[/yellow]")
            return
        
        table = Table(title="Jobs")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Queue", style="yellow")
        table.add_column("Config Preview", style="dim")
        
        for job in job_list:
            config_preview = job.config_text[:40] + "..." if len(job.config_text) > 40 else job.config_text
            # Shorten queue URL for display
            queue_short = job.queue_url.split('/')[-1] if '/' in job.queue_url else job.queue_url
            table.add_row(str(job.id), job.name, queue_short, config_preview)
        
        console.print(table)
    finally:
        conn.close()


@jobs.command()
@click.argument('job_id', type=int)
def show(job_id):
    """Show job details."""
    from bec_orch.orch.job_admin import get_job
    
    conn = get_db_connection()
    try:
        job = get_job(conn, job_id)
        
        console.print(f"\n[bold]Job {job.id}[/bold]")
        console.print(f"Name: [green]{job.name}[/green]")
        console.print(f"Queue URL: [yellow]{job.queue_url}[/yellow]")
        if job.dlq_url:
            console.print(f"DLQ URL: [yellow]{job.dlq_url}[/yellow]")
        console.print(f"\nConfiguration:")
        
        # Pretty print config if it's JSON
        try:
            config_obj = json.loads(job.config_text)
            console.print_json(data=config_obj)
        except json.JSONDecodeError:
            console.print(job.config_text)
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        conn.close()


@jobs.command()
@click.argument('job_id', type=int)
@click.option('--config', type=click.Path(exists=True), help='Path to config file (JSON)')
@click.option('--config-text', help='Config text directly (JSON string)')
def update(job_id, config, config_text):
    """Update job configuration."""
    from bec_orch.orch.job_admin import update_job_config
    
    if not config and not config_text:
        console.print("[red]Error:[/red] Must specify either --config or --config-text")
        sys.exit(1)
    
    if config and config_text:
        console.print("[red]Error:[/red] Cannot specify both --config and --config-text")
        sys.exit(1)
    
    if config:
        config_text = Path(config).read_text()
    
    # Validate JSON
    try:
        json.loads(config_text)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        sys.exit(1)
    
    conn = get_db_connection()
    try:
        update_job_config(conn, job_id, config_text)
        console.print(f"[green]✓[/green] Updated job {job_id}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        conn.close()


@jobs.command()
@click.argument('job_id', type=int)
@click.option('--yes', is_flag=True, help='Skip confirmation')
def delete(job_id, yes):
    """Delete a job (and all its task executions)."""
    from bec_orch.orch.job_admin import delete_job
    
    if not yes:
        confirm = click.confirm(
            f"Are you sure you want to delete job {job_id}? This will also delete all task executions.",
            abort=True
        )
    
    conn = get_db_connection()
    try:
        delete_job(conn, job_id)
        console.print(f"[green]✓[/green] Deleted job {job_id}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        conn.close()


@jobs.command()
@click.argument('job_name')
def stats(job_name):
    """Show job statistics (task execution summary)."""
    from bec_orch.orch.job_admin import get_job_by_name
    from tabulate import tabulate
    
    conn = get_db_connection()
    try:
        # Get job
        job = get_job_by_name(conn, job_name)
        
        # Get statistics
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    status,
                    COUNT(*) as count,
                    ROUND(AVG(total_duration_ms)::numeric, 2) as avg_duration_ms,
                    SUM(total_images) as total_images,
                    SUM(nb_errors) as total_errors
                FROM task_executions 
                WHERE job_id = %s 
                GROUP BY status
                ORDER BY 
                    CASE status
                        WHEN 'running' THEN 1
                        WHEN 'done' THEN 2
                        WHEN 'retryable_failed' THEN 3
                        WHEN 'terminal_failed' THEN 4
                    END
                """,
                (job.id,)
            )
            stats_rows = cur.fetchall()
            
            # Get overall totals
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(total_images) as total_images,
                    SUM(nb_errors) as total_errors,
                    ROUND(AVG(total_duration_ms)::numeric, 2) as avg_duration_ms,
                    ROUND(AVG(avg_duration_per_page_ms)::numeric, 2) as avg_per_page_ms
                FROM task_executions 
                WHERE job_id = %s
                """,
                (job.id,)
            )
            totals = cur.fetchone()
        
        console.print(f"\n[bold]Job Statistics: {job.name}[/bold] (id={job.id})\n")
        
        if not stats_rows:
            console.print("[yellow]No task executions found[/yellow]")
            return
        
        # Status breakdown table
        table = Table(title="Task Execution Status")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Total Images", justify="right")
        table.add_column("Total Errors", justify="right")
        table.add_column("Avg Duration (ms)", justify="right")
        
        for row in stats_rows:
            status_style = {
                'running': 'yellow',
                'done': 'green',
                'retryable_failed': 'orange1',
                'terminal_failed': 'red'
            }.get(row['status'], 'white')
            
            table.add_row(
                f"[{status_style}]{row['status']}[/{status_style}]",
                str(row['count']),
                str(row['total_images'] or 0),
                str(row['total_errors'] or 0),
                str(row['avg_duration_ms'] or 0)
            )
        
        console.print(table)
        
        # Overall totals
        console.print(f"\n[bold]Overall Totals:[/bold]")
        console.print(f"  Total Tasks: {totals['total_tasks']}")
        console.print(f"  Total Images: {totals['total_images'] or 0}")
        console.print(f"  Total Errors: {totals['total_errors'] or 0}")
        if totals['avg_duration_ms']:
            console.print(f"  Avg Duration per Task: {totals['avg_duration_ms']:.2f} ms")
        if totals['avg_per_page_ms']:
            console.print(f"  Avg Duration per Page: {totals['avg_per_page_ms']:.2f} ms")
        console.print()
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        conn.close()


@jobs.command()
@click.argument('job_name')
@click.option('--limit', type=int, default=10, help='Number of recent tasks to show (default: 10)')
@click.option('--status', help='Filter by status (running, done, retryable_failed, terminal_failed)')
def tasks(job_name, limit, status):
    """Show recent task executions for a job."""
    from bec_orch.orch.job_admin import get_job_by_name
    
    conn = get_db_connection()
    try:
        # Get job
        job = get_job_by_name(conn, job_name)
        
        # Build query
        query = """
            SELECT 
                te.id,
                v.bdrc_w_id,
                v.bdrc_i_id,
                te.status,
                te.started_at,
                te.done_at,
                te.total_images,
                te.nb_errors,
                te.total_duration_ms,
                w.worker_name
            FROM task_executions te
            JOIN volumes v ON te.volume_id = v.id
            LEFT JOIN workers w ON te.worker_id = w.worker_id
            WHERE te.job_id = %s
        """
        params = [job.id]
        
        if status:
            query += " AND te.status = %s"
            params.append(status)
        
        query += " ORDER BY te.id DESC LIMIT %s"
        params.append(limit)
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            tasks_rows = cur.fetchall()
        
        console.print(f"\n[bold]Recent Task Executions: {job.name}[/bold] (id={job.id})")
        if status:
            console.print(f"[dim]Filtered by status: {status}[/dim]")
        console.print()
        
        if not tasks_rows:
            console.print("[yellow]No task executions found[/yellow]")
            return
        
        # Create table
        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Volume", style="green")
        table.add_column("Status")
        table.add_column("Images", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Worker", style="dim")
        
        for row in tasks_rows:
            status_style = {
                'running': 'yellow',
                'done': 'green',
                'retryable_failed': 'orange1',
                'terminal_failed': 'red'
            }.get(row['status'], 'white')
            
            duration_s = f"{row['total_duration_ms'] / 1000:.1f}" if row['total_duration_ms'] else "-"
            worker_name = row['worker_name'][:15] if row['worker_name'] else "-"
            
            table.add_row(
                str(row['id']),
                f"{row['bdrc_w_id']}/{row['bdrc_i_id']}",
                f"[{status_style}]{row['status']}[/{status_style}]",
                str(row['total_images'] or "-"),
                str(row['nb_errors'] or "-"),
                duration_s,
                worker_name
            )
        
        console.print(table)
        console.print()
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        conn.close()

