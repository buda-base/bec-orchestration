#!/usr/bin/env python3
"""
BEC Orchestration CLI - Main entry point.

Commands:
  bec worker    - Run worker to process tasks
  bec jobs      - Manage jobs (create, show, update)
  bec queue     - Manage task queues (enqueue, stats, redrive)
"""

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from bec_orch import __version__

# Setup rich console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """BEC Orchestration - Large-scale task processing on AWS."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


# Import subcommands
from bec_orch.cli.worker import worker
from bec_orch.cli.jobs import jobs
from bec_orch.cli.queue import queue

cli.add_command(worker)
cli.add_command(jobs)
cli.add_command(queue)


def main():
    """Main entry point."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()

