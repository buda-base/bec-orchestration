"""
BEC Queue CLI - Manage task queues.

Usage:
  bec queue enqueue --queue-url <url> --file volumes.txt
  bec queue enqueue --queue-url <url> --volume W12345,I0123
  bec queue enqueue --job-name ldv1 --volume W12345,I0123
  bec queue stats --queue-url <url>
  bec queue stats --job-name ldv1
  bec queue purge --queue-url <url>
  bec queue purge --job-name ldv1
"""

import os
import sys
from datetime import datetime
from urllib.parse import quote_plus

import click
from rich.console import Console
from rich.table import Table

console = Console()


def get_sqs_client():
    """Get SQS client from environment."""
    from bec_orch.io.sqs import SQSClient
    
    region = os.environ.get('BEC_REGION', 'us-east-1')
    return SQSClient(region)


def get_queue_url_from_job_name(job_name: str) -> str:
    """Get queue URL from job name by querying the database."""
    from bec_orch.io.db import DBClient, build_dsn_from_env
    from bec_orch.orch.job_admin import get_job_by_name
    
    try:
        dsn = build_dsn_from_env()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    db = DBClient(dsn)
    conn = db.connect()
    try:
        job = get_job_by_name(conn, job_name)
        return job.queue_url
    finally:
        conn.close()


@click.group()
def queue():
    """Manage task queues (enqueue, stats, purge)."""
    pass


@queue.command()
@click.option('--queue-url', help='SQS queue URL')
@click.option('--job-name', help='Job name (alternative to --queue-url)')
@click.option('--file', type=click.Path(exists=True), help='File with volume list (one per line: W12345,I0123)')
@click.option('--volume', multiple=True, help='Single volume (format: W12345,I0123). Can be specified multiple times.')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
@click.option('-n', '--limit', type=int, help='Limit number of volumes to enqueue (after filtering)')
@click.option('-f', '--force', is_flag=True, help='Force enqueue all volumes, ignoring done status in DB')
def enqueue(queue_url, job_name, file, volume, region, limit, force):
    """Enqueue volumes to task queue."""
    from bec_orch.core.models import VolumeRef
    from bec_orch.orch.enqueue import enqueue_volumes, enqueue_volume_list_from_file
    from bec_orch.io.db import DBClient, build_dsn_from_env
    from bec_orch.orch.job_admin import get_job_by_name
    
    # Get queue URL from job name if provided
    job_id = None
    if job_name:
        if queue_url:
            console.print("[red]Error:[/red] Cannot specify both --queue-url and --job-name")
            sys.exit(1)
        queue_url = get_queue_url_from_job_name(job_name)
        console.print(f"[dim]Using queue URL from job '{job_name}': {queue_url}[/dim]")
        
        # Get job_id for filtering (if not forcing)
        if not force:
            try:
                dsn = build_dsn_from_env()
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("Or use --force to skip filtering")
                sys.exit(1)
            
            db = DBClient(dsn)
            conn = db.connect()
            try:
                job = get_job_by_name(conn, job_name)
                job_id = job.id
            finally:
                conn.close()
    elif not queue_url:
        console.print("[red]Error:[/red] Must specify either --queue-url or --job-name")
        sys.exit(1)
    
    if not file and not volume:
        console.print("[red]Error:[/red] Must specify either --file or --volume")
        sys.exit(1)
    
    if file and volume:
        console.print("[red]Error:[/red] Cannot specify both --file and --volume")
        sys.exit(1)
    
    # Validate limit
    if limit is not None and limit <= 0:
        console.print("[red]Error:[/red] Limit must be a positive integer")
        sys.exit(1)
    
    # Override region if provided
    if region:
        os.environ['BEC_REGION'] = region
    
    sqs = get_sqs_client()
    
    try:
        if file:
            console.print(f"Enqueueing volumes from file: [cyan]{file}[/cyan]")
            
            # Set up filtering if not forcing and job_id is available
            filter_func = None
            conn = None
            if not force and job_id is not None:
                console.print("[dim]Fetching volumes already done on latest version...[/dim]")
                
                try:
                    dsn = build_dsn_from_env()
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    sys.exit(1)
                
                db = DBClient(dsn)
                conn = db.connect()
                
                # Fetch all done volumes in one query (efficient for large lists)
                done_volumes = db.get_volumes_done_on_latest_version(conn, job_id)
                console.print(f"[dim]Found {len(done_volumes)} volume(s) already done on latest version[/dim]")
                
                def should_enqueue(vol: VolumeRef) -> bool:
                    """Return True if volume should be enqueued, False if already done on latest version."""
                    return (vol.w_id, vol.i_id) not in done_volumes
                
                filter_func = should_enqueue
            
            try:
                enqueued_count, skipped_count = enqueue_volume_list_from_file(
                    sqs, queue_url, file, filter_func=filter_func, limit=limit
                )
                
                if filter_func is not None:
                    console.print(f"[green]✓[/green] Enqueued {enqueued_count} volume(s), skipped {skipped_count} already done")
                else:
                    console.print(f"[green]✓[/green] Enqueued {enqueued_count} volume(s)")
            finally:
                if conn is not None:
                    conn.close()
        else:
            # Parse volume arguments
            volumes = []
            for vol_str in volume:
                parts = vol_str.replace(',', ' ').split()
                if len(parts) != 2:
                    console.print(f"[red]Error:[/red] Invalid volume format: {vol_str} (expected W12345,I0123)")
                    sys.exit(1)
                volumes.append(VolumeRef(w_id=parts[0], i_id=parts[1]))
            
            console.print(f"Enqueueing {len(volumes)} volume(s)...")
            count = enqueue_volumes(sqs, queue_url, volumes)
            console.print(f"[green]✓[/green] Enqueued {count} volume(s)")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@queue.command()
@click.option('--queue-url', help='SQS queue URL')
@click.option('--job-name', help='Job name (alternative to --queue-url)')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def stats(queue_url, job_name, region):
    """Show queue statistics."""
    from bec_orch.orch.enqueue import get_queue_stats
    
    # Get queue URL from job name if provided
    if job_name:
        if queue_url:
            console.print("[red]Error:[/red] Cannot specify both --queue-url and --job-name")
            sys.exit(1)
        queue_url = get_queue_url_from_job_name(job_name)
        console.print(f"[dim]Using queue URL from job '{job_name}': {queue_url}[/dim]")
    elif not queue_url:
        console.print("[red]Error:[/red] Must specify either --queue-url or --job-name")
        sys.exit(1)
    
    # Override region if provided
    if region:
        os.environ['BEC_REGION'] = region
    
    sqs = get_sqs_client()
    
    try:
        stats_data = get_queue_stats(sqs, queue_url)
        
        table = Table(title=f"Queue Stats: {queue_url}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Messages Available", str(stats_data['approximate_messages']))
        table.add_row("Messages In Flight", str(stats_data['approximate_messages_not_visible']))
        table.add_row("Messages Delayed", str(stats_data['approximate_messages_delayed']))
        
        # Convert timestamps
        created = datetime.fromtimestamp(stats_data['created_timestamp'])
        modified = datetime.fromtimestamp(stats_data['last_modified_timestamp'])
        
        table.add_row("Created", created.strftime('%Y-%m-%d %H:%M:%S'))
        table.add_row("Last Modified", modified.strftime('%Y-%m-%d %H:%M:%S'))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@queue.command()
@click.option('--queue-url', help='SQS queue URL')
@click.option('--job-name', help='Job name (alternative to --queue-url)')
@click.option('--yes', is_flag=True, help='Skip confirmation')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def purge(queue_url, job_name, yes, region):
    """Purge all messages from queue (WARNING: irreversible!)."""
    from bec_orch.orch.enqueue import purge_queue
    
    # Get queue URL from job name if provided
    if job_name:
        if queue_url:
            console.print("[red]Error:[/red] Cannot specify both --queue-url and --job-name")
            sys.exit(1)
        queue_url = get_queue_url_from_job_name(job_name)
        console.print(f"[dim]Using queue URL from job '{job_name}': {queue_url}[/dim]")
    elif not queue_url:
        console.print("[red]Error:[/red] Must specify either --queue-url or --job-name")
        sys.exit(1)
    
    if not yes:
        console.print("[yellow]WARNING:[/yellow] This will delete all messages from the queue!")
        confirm = click.confirm("Are you sure?", abort=True)
    
    # Override region if provided
    if region:
        os.environ['BEC_REGION'] = region
    
    sqs = get_sqs_client()
    
    try:
        purge_queue(sqs, queue_url)
        console.print(f"[green]✓[/green] Purged queue")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@queue.command()
@click.option('--source-queue-url', required=True, help='Source queue URL (e.g., DLQ)')
@click.option('--dest-queue-url', required=True, help='Destination queue URL')
@click.option('--max-messages', type=int, default=1000, help='Maximum messages to redrive')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def redrive(source_queue_url, dest_queue_url, max_messages, region):
    """Redrive messages from DLQ back to main queue."""
    from bec_orch.io.sqs import SQSClient
    
    # Override region if provided
    if region:
        os.environ['BEC_REGION'] = region
    
    sqs = get_sqs_client()
    
    console.print(f"Redriving messages from DLQ to main queue...")
    console.print(f"Source: {source_queue_url}")
    console.print(f"Dest: {dest_queue_url}")
    
    count = 0
    
    try:
        while count < max_messages:
            # Receive message from source (DLQ)
            msg = sqs.receive_one(source_queue_url, wait_seconds=1, visibility_timeout=30)
            
            if msg is None:
                break
            
            # Send to destination queue
            sqs.send_raw(dest_queue_url, msg.body, w_id=msg.volume.w_id, i_id=msg.volume.i_id)
            
            # Delete from source
            sqs.delete(source_queue_url, msg.receipt_handle)
            
            count += 1
            
            if count % 10 == 0:
                console.print(f"Redriven {count} messages...")
        
        console.print(f"[green]✓[/green] Redriven {count} message(s)")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

