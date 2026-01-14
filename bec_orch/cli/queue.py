"""
BEC Queue CLI - Manage task queues.

Usage:
  bec queue enqueue --queue-url <url> --file volumes.txt
  bec queue enqueue --queue-url <url> --volume W12345,I0123
  bec queue stats --queue-url <url>
  bec queue purge --queue-url <url>
"""

import os
import sys
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table

console = Console()


def get_sqs_client():
    """Get SQS client from environment."""
    from bec_orch.io.sqs import SQSClient
    
    region = os.environ.get('BEC_REGION', 'us-east-1')
    return SQSClient(region)


@click.group()
def queue():
    """Manage task queues (enqueue, stats, purge)."""
    pass


@queue.command()
@click.option('--queue-url', required=True, help='SQS queue URL')
@click.option('--file', type=click.Path(exists=True), help='File with volume list (one per line: W12345,I0123)')
@click.option('--volume', multiple=True, help='Single volume (format: W12345,I0123). Can be specified multiple times.')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def enqueue(queue_url, file, volume, region):
    """Enqueue volumes to task queue."""
    from bec_orch.core.models import VolumeRef
    from bec_orch.orch.enqueue import enqueue_volumes, enqueue_volume_list_from_file
    
    if not file and not volume:
        console.print("[red]Error:[/red] Must specify either --file or --volume")
        sys.exit(1)
    
    if file and volume:
        console.print("[red]Error:[/red] Cannot specify both --file and --volume")
        sys.exit(1)
    
    # Override region if provided
    if region:
        os.environ['BEC_REGION'] = region
    
    sqs = get_sqs_client()
    
    try:
        if file:
            console.print(f"Enqueueing volumes from file: [cyan]{file}[/cyan]")
            count = enqueue_volume_list_from_file(sqs, queue_url, file)
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
        sys.exit(1)


@queue.command()
@click.option('--queue-url', required=True, help='SQS queue URL')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def stats(queue_url, region):
    """Show queue statistics."""
    from bec_orch.orch.enqueue import get_queue_stats
    
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
@click.option('--queue-url', required=True, help='SQS queue URL')
@click.option('--yes', is_flag=True, help='Skip confirmation')
@click.option('--region', help='AWS region (default: from BEC_REGION env or us-east-1)')
def purge(queue_url, yes, region):
    """Purge all messages from queue (WARNING: irreversible!)."""
    from bec_orch.orch.enqueue import purge_queue
    
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

