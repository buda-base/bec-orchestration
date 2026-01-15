"""
BEC Worker CLI - Run worker instances to process tasks.

Usage:
  bec worker --job-id 1 --queue-url https://sqs...
  
Environment variables:
  BEC_SQL_HOST, BEC_SQL_PORT, BEC_SQL_USER, BEC_SQL_PASSWORD
  BEC_REGION (AWS region)
  BEC_DEST_S3_BUCKET (destination bucket for artifacts)
  
For ldv1 job:
  BEC_LD_MODEL_PATH (path to .pth model file)
"""

import logging
import os
import sys
from urllib.parse import quote_plus

import click
from dotenv import load_dotenv
from rich.console import Console

console = Console()
# Use "bec" namespace so logs appear at INFO level
logger = logging.getLogger("bec.cli.worker")

# Load environment variables from .env file if present
load_dotenv()


@click.command()
@click.option('--job-name', type=str, required=True, help='Job name from database (e.g., ldv1, ocr)')
@click.option('--poll-wait', type=int, default=20, help='SQS long-poll wait time (seconds)')
@click.option('--visibility-timeout', type=int, default=300, help='SQS visibility timeout (seconds)')
@click.option('--shutdown-after-empty', type=int, default=6, help='Shutdown after N empty polls')
@click.option('--s3-source-bucket', type=str, default='archive.tbrc.org', help='Source S3 bucket for images')
@click.option('--s3-dest-bucket', type=str, help='Destination S3 bucket (default: from BEC_DEST_S3_BUCKET env)')
@click.option('--region', type=str, help='AWS region (default: from BEC_REGION env or us-east-1)')
@click.option('--model-path', type=str, help='Path to model file (for ML jobs, default: from BEC_LD_MODEL_PATH env)')
def worker(
    job_name,
    poll_wait,
    visibility_timeout,
    shutdown_after_empty,
    s3_source_bucket,
    s3_dest_bucket,
    region,
    model_path,
):
    """Run worker to process tasks from SQS queue.
    
    Queue URLs are fetched from the job record in the database.
    """
    
    # Setup logging FIRST: root=WARNING, bec namespace=INFO
    # This ensures all subsequent logging calls use JSON format
    from bec_orch.logging_setup import setup_logging
    setup_logging()
    
    from bec_orch.config import OrchestrationConfig
    from bec_orch.core.worker_runtime import BECWorkerRuntime
    from bec_orch.io.db import DBClient
    from bec_orch.io.sqs import SQSClient
    
    # Build DSN from environment
    sql_host = os.environ.get('BEC_SQL_HOST')
    sql_port = os.environ.get('BEC_SQL_PORT', '5432')
    sql_user = os.environ.get('BEC_SQL_USER')
    sql_password = os.environ.get('BEC_SQL_PASSWORD')
    sql_database = os.environ.get('BEC_SQL_DATABASE', 'pipeline_v1')
    
    if not all([sql_host, sql_user, sql_password]):
        logger.error("Missing required SQL environment variables: BEC_SQL_HOST, BEC_SQL_USER, BEC_SQL_PASSWORD")
        sys.exit(1)
    
    # URL-encode password to handle special characters
    encoded_password = quote_plus(sql_password)
    # Add SSL requirement for secure connections
    dsn = f"postgresql://{sql_user}:{encoded_password}@{sql_host}:{sql_port}/{sql_database}?sslmode=require"
    
    # Get AWS region
    region = region or os.environ.get('BEC_REGION', 'us-east-1')
    
    # Get destination bucket
    s3_dest_bucket = s3_dest_bucket or os.environ.get('BEC_DEST_S3_BUCKET')
    if not s3_dest_bucket:
        logger.error("Missing destination S3 bucket. Provide --s3-dest-bucket or set BEC_DEST_S3_BUCKET environment variable")
        sys.exit(1)
    
    # Set model path in environment if provided (for ldv1 worker)
    if model_path:
        os.environ['BEC_LD_MODEL_PATH'] = model_path
    elif 'BEC_LD_MODEL_PATH' not in os.environ:
        # Try to infer from job name if it's an LD job
        logger.warning("BEC_LD_MODEL_PATH not set. If this is an LD job, provide --model-path or set env var.")
    
    # Create config
    cfg = OrchestrationConfig(
        db_dsn=dsn,
        aws_region=region,
        job_name=job_name,
        poll_wait_seconds=poll_wait,
        visibility_timeout_seconds=visibility_timeout,
        shutdown_after_empty_polls=shutdown_after_empty,
    )
    
    logger.info("Starting BEC Worker", extra={"job_name": job_name, "region": region, "s3_source_bucket": s3_source_bucket, "s3_dest_bucket": s3_dest_bucket})
    
    # Create clients
    db = DBClient(cfg.db_dsn)
    sqs = SQSClient(cfg.aws_region)
    
    # Create and run worker runtime
    runtime = BECWorkerRuntime(
        cfg=cfg,
        db=db,
        sqs=sqs,
        s3_source_bucket=s3_source_bucket,
        s3_dest_bucket=s3_dest_bucket,
    )
    
    try:
        runtime.initialize()
        logger.info("Worker initialized", extra={"worker_id": runtime.worker_id, "job_id": runtime.job_record.id, "queue_url": runtime.queue_url, "dlq_url": runtime.dlq_url})
        runtime.run_forever()
        logger.info("Worker completed successfully")
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        raise
    finally:
        runtime.shutdown()
        logger.info("Worker shutdown complete")

