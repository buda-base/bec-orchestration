# BEC Orchestration CLI Guide

## Installation

```bash
# Install core + CLI dependencies
pip install -e .

# For ldv1 worker (with ML dependencies)
pip install -e ".[ldv1]"

# Note: PyTorch should be installed separately or use AWS DLAMI pre-installed version
```

## Environment Setup

Create a `.env` file or export these variables:

```bash
# Database (required)
export BEC_SQL_HOST="your-postgres-host.rds.amazonaws.com"
export BEC_SQL_PORT="5432"
export BEC_SQL_USER="postgres"
export BEC_SQL_PASSWORD="your-password"
export BEC_SQL_DATABASE="pipeline_v1"  # optional, defaults to pipeline_v1

# AWS (required)
export BEC_REGION="us-east-1"
export BEC_DEST_S3_BUCKET="your-artifacts-bucket"

# For ldv1 job (required when running ldv1 worker)
export BEC_LD_MODEL_PATH="/path/to/model.pth"
```

## Queue Naming Convention

BEC automatically creates SQS queues when you create a job, following this convention:

```
Queue:  bec_{job_name}_tasks
DLQ:    bec_{job_name}_dlq

Examples:
- ldv1 → bec_ldv1_tasks / bec_ldv1_dlq
```

**No manual queue management required!** The CLI creates queues automatically.

## CLI Commands

### Job Management

#### Create a job

Creating a job **automatically creates the SQS queues**:

```bash
# Simple - queues are created automatically!
bec jobs create --name ldv1

# With config from file
bec jobs create --name ldv1 --config ldv1-config.json

# With inline config
bec jobs create --name ocr --config-text '{"model": "tesseract", "lang": "bod"}'
```

Output:
```
Creating SQS queues for job 'ldv1'...
✓ Created queue: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks
✓ Created DLQ: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq
✓ Created job: id=1, name=ldv1
```

Example config file (`ldv1-config.json`):
```json
{
  "use_gpu": true,
  "precision": "fp16",
  "batch_size": 16,
  "compile_model": false,
  "max_tiles_per_batch": 80
}
```

**Note:** Model path is NOT in the config file. It comes from `BEC_LD_MODEL_PATH` env var or `--model-path` CLI argument.

**Advanced:** Use existing queues with custom URLs:
```bash
bec jobs create --name ldv1 \
  --queue-url https://sqs.us-east-1.amazonaws.com/.../custom-queue \
  --dlq-url https://sqs.us-east-1.amazonaws.com/.../custom-dlq \
  --config ldv1-config.json
```

#### List all jobs

```bash
bec jobs list
```

Output:
```
┏━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID ┃ Name ┃ Queue            ┃ Config       ┃
┡━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1  │ ldv1 │ bec_ldv1_tasks   │ {"use_gp...  │
└────┴──────┴──────────────────┴──────────────┘
```

#### Show job details

```bash
bec jobs show ldv1
```

Output:
```
Job 1
Name: ldv1
Queue URL: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks
DLQ URL: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq
Configuration:
{
  "use_gpu": true,
  "batch_size": 16
}
```

#### Update job config

```bash
bec jobs update ldv1 --config ldv1-config.json
```

#### Delete a job

```bash
bec jobs delete ldv1 --yes
```

#### Show job statistics

View task execution statistics for a job:

```bash
bec jobs stats ldv1
```

Output:
```
Job Statistics: ldv1 (id=1)

         Task Execution Status          
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Status            ┃ Count ┃ Total Images┃ Total Errors┃ Avg Duration (ms) ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ running           │    2  │       400   │      0      │        0          │
│ done              │   45  │      9000   │     12      │     15234.56      │
│ retryable_failed  │    1  │       200   │     15      │     12000.00      │
│ terminal_failed   │    2  │       400   │     200     │      8500.00      │
└───────────────────┴───────┴─────────────┴─────────────┴───────────────────┘

Overall Totals:
  Total Tasks: 50
  Total Images: 10000
  Total Errors: 227
  Avg Duration per Task: 14567.89 ms
  Avg Duration per Page: 145.67 ms
```

#### Show recent task executions

View recent task executions:

```bash
# Show last 10 tasks (default)
bec jobs tasks ldv1

# Show last 20 tasks
bec jobs tasks ldv1 --limit 20

# Filter by status
bec jobs tasks ldv1 --status done
bec jobs tasks ldv1 --status retryable_failed --limit 50
```

Output:
```
Recent Task Executions: ldv1 (id=1)

┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ ID   ┃ Volume             ┃ Status           ┃ Images ┃ Errors ┃ Duration (s)┃ Worker        ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 105  │ W22084/I0886       │ done             │    200 │      2 │        15.2 │ i-abc123      │
│ 104  │ W22084/I0887       │ done             │    198 │      0 │        14.8 │ i-abc123      │
│ 103  │ W1KG13780/I1KG...  │ running          │      - │      - │           - │ i-def456      │
│ 102  │ W22084/I0888       │ retryable_failed │    150 │     15 │        10.5 │ i-abc123      │
└──────┴────────────────────┴──────────────────┴────────┴────────┴─────────────┴───────────────┘
```

### Queue Management

#### Enqueue volumes

The CLI automatically uses the correct queue for the job - just specify the job name!

From a file:
```bash
bec queue enqueue --job-name ldv1 --file volumes.txt
```

Volume file format (`volumes.txt`):
```
# One volume per line: W_ID,I_ID or W_ID I_ID
W22084,I0886
W22084 I0887
W1KG13780,I1KG15541

# Comments start with #
# Empty lines are ignored
```

Single volume:
```bash
bec queue enqueue --job-name ldv1 --volume W22084,I0886
```

Multiple volumes:
```bash
bec queue enqueue --job-name ldv1 \
  --volume W22084,I0886 \
  --volume W22084,I0887
```

#### Queue statistics

```bash
bec queue stats --job-name ldv1
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                    ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Messages Available        │ 1234    │
│ Messages In Flight        │ 5       │
│ Messages Delayed          │ 0       │
│ Created                   │ 2026... │
│ Last Modified             │ 2026... │
└───────────────────────────┴─────────┘
```

#### Purge queue (delete all messages)

```bash
bec queue purge --job-name ldv1 --yes
```

⚠️ **WARNING:** This is irreversible!

#### Redrive messages from DLQ

Move messages from dead-letter queue back to main queue:

```bash
# Get queue URLs from job first
JOB_INFO=$(bec jobs show ldv1)

# Then redrive
bec queue redrive \
  --source-queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq \
  --dest-queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks \
  --max-messages 1000
```

Or use AWS CLI directly:
```bash
aws sqs start-message-move-task \
  --source-arn arn:aws:sqs:us-east-1:123456789:bec_ldv1_dlq \
  --destination-arn arn:aws:sqs:us-east-1:123456789:bec_ldv1_tasks
```

### Worker

#### Run a worker

Workers automatically use the queue from the job - just specify the job name!

```bash
bec worker --job-name ldv1 --model-path /path/to/model.pth
```

Output:
```
Starting BEC Worker
Job: ldv1
Region: us-east-1
Source bucket: archive.tbrc.org
Dest bucket: artifacts-bucket
Worker initialized: worker_id=1, job_id=1
Queue: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks
DLQ: https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq
Processing message: ... for volume W22084/I0886
Successfully processed message: ...
...
Worker completed successfully
Worker shutdown complete
```

#### Worker options

```bash
bec worker --job-name ldv1 \
  --poll-wait 20 \
  --visibility-timeout 300 \
  --shutdown-after-empty 6 \
  --s3-source-bucket archive.tbrc.org \
  --s3-dest-bucket artifacts.example.com \
  --region us-east-1 \
  --model-path /path/to/model.pth
```

Options:
- `--job-name`: Job name (required)
- `--poll-wait`: SQS long-poll wait time in seconds (default: 20)
- `--visibility-timeout`: SQS visibility timeout in seconds (default: 300)
- `--shutdown-after-empty`: Shutdown after N empty polls (default: 6)
- `--s3-source-bucket`: Source bucket for images (default: archive.tbrc.org)
- `--s3-dest-bucket`: Destination bucket for artifacts (from env if not specified)
- `--region`: AWS region (from BEC_REGION env if not specified)
- `--model-path`: Path to model file (from BEC_LD_MODEL_PATH env if not specified)

#### Worker behavior

1. **Initialization:**
   - Connects to database
   - Registers as a worker with heartbeat
   - Fetches job configuration and queue URLs
   - Loads job worker implementation

2. **Main loop:**
   - Polls SQS for messages (long-polling)
   - For each message:
     - Fetches volume manifest from S3
     - Ensures volume exists in DB
     - Claims task (idempotent)
     - Checks for existing success marker
     - Runs job worker
     - Writes success marker to S3
     - Updates DB with metrics
     - Deletes SQS message
   - Tracks empty polls
   - Shuts down after N empty polls (configurable)

3. **Shutdown:**
   - Marks worker as stopped in DB
   - Closes connections gracefully

## Quick Start Guide

### 1. Setup Environment

```bash
# Create .env file
cat > .env <<EOF
BEC_SQL_HOST=postgres.example.com
BEC_SQL_USER=pipeline_user
BEC_SQL_PASSWORD=secret
BEC_REGION=us-east-1
BEC_DEST_S3_BUCKET=bdrc-artifacts
BEC_LD_MODEL_PATH=/models/line-detection-v1.pth
EOF

# Load environment
export $(cat .env | xargs)
```

### 2. Create Database Schema

```bash
psql -h $BEC_SQL_HOST -U $BEC_SQL_USER -d pipeline_v1 -f schema.sql
```

### 3. Create a Job

```bash
# Create job (SQS queues created automatically!)
bec jobs create --name ldv1 --config-text '{
  "use_gpu": true,
  "precision": "fp16",
  "batch_size": 16
}'
```

### 4. Enqueue Volumes

```bash
# Create volume list
cat > volumes.txt <<EOF
W22084,I0886
W22084,I0887
W1KG13780,I1KG15541
EOF

# Enqueue volumes
bec queue enqueue --job-name ldv1 --file volumes.txt
```

### 5. Start Worker

```bash
# Run worker
bec worker --job-name ldv1
```

### 6. Monitor Progress

Check queue stats:
```bash
bec queue stats --job-name ldv1
```

Check job statistics:
```bash
# Overall statistics
bec jobs stats ldv1

# Recent task executions
bec jobs tasks ldv1

# Filter by status
bec jobs tasks ldv1 --status done
bec jobs tasks ldv1 --status retryable_failed --limit 50
```

## Running Multiple Workers

Start multiple workers in parallel to process tasks faster:

```bash
# Terminal 1
bec worker --job-name ldv1

# Terminal 2
bec worker --job-name ldv1

# Terminal 3
bec worker --job-name ldv1
```

Each worker will claim different tasks from the same queue (via idempotent DB claiming).

## Systemd Service

Deploy worker as systemd service (see `bec-worker.service`):

```bash
sudo cp bec-worker.service /etc/systemd/system/
sudo systemctl enable bec-worker
sudo systemctl start bec-worker
sudo systemctl status bec-worker
```

## Using with Screen

```bash
# Start screen session
screen -S bec-worker

# Run worker
bec worker --job-name ldv1

# Detach: Ctrl+A, D
# Reattach: screen -r bec-worker
```

## Troubleshooting

### Worker won't start

Check environment variables:
```bash
echo $BEC_SQL_HOST
echo $BEC_DEST_S3_BUCKET
echo $BEC_LD_MODEL_PATH
```

Test database connection:
```bash
psql -h $BEC_SQL_HOST -U $BEC_SQL_USER -d pipeline_v1 -c "SELECT 1"
```

Test SQS access:
```bash
aws sqs list-queues --queue-name-prefix bec_ --region $BEC_REGION
```

### Tasks failing

Check DLQ:
```bash
# Get DLQ stats
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq \
  --attribute-names ApproximateNumberOfMessages
```

Redrive failed messages:
```bash
bec queue redrive \
  --source-queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_dlq \
  --dest-queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks
```

View failed tasks in database:
```sql
SELECT * FROM task_executions 
WHERE job_id = 1 AND status IN ('retryable_failed', 'terminal_failed')
ORDER BY done_at DESC;
```

### Model not found

Ensure model path is correct:
```bash
ls -lh $BEC_LD_MODEL_PATH
```

Set explicitly:
```bash
export BEC_LD_MODEL_PATH=/absolute/path/to/model.pth
bec worker --job-name ldv1 --model-path $BEC_LD_MODEL_PATH
```

### Queue already exists

If you get an error that the queue already exists, the CLI will reuse it automatically. No action needed.

### Permission denied

Ensure your AWS credentials have these SQS permissions:
- `sqs:CreateQueue`
- `sqs:GetQueueUrl`
- `sqs:GetQueueAttributes`
- `sqs:SetQueueAttributes`
- `sqs:SendMessage`
- `sqs:ReceiveMessage`
- `sqs:DeleteMessage`
- `sqs:ChangeMessageVisibility`

## Best Practices

1. **Always use environment variables for secrets** (passwords, keys)
2. **Use DLQ for failed tasks** (automatically configured)
3. **Monitor queue depth** to scale workers up/down
4. **Set appropriate visibility timeout** (should be > max task duration)
5. **Use meaningful job names** for easy identification
6. **Keep job config in version control** (JSON files)
7. **Test with small batches first** before enqueueing thousands of volumes
8. **Monitor worker heartbeats** to detect stuck workers
9. **Use systemd for production** to auto-restart on failures
10. **Check CloudWatch logs** for detailed error messages

## Advanced Topics

### Creating Multiple Jobs

```bash
# Each job gets its own queues automatically
bec jobs create --name ldv1 --config ldv1.json
bec jobs create --name ldv2 --config ldv2.json
bec jobs create --name ocr --config ocr.json

# Queues created:
# - bec_ldv1_tasks / bec_ldv1_dlq
# - bec_ldv2_tasks / bec_ldv2_dlq
# - bec_ocr_tasks / bec_ocr_dlq
```

### Viewing Queue URLs

```bash
# Show job details to see queue URLs
bec jobs show ldv1

# Or list all jobs with queue names
bec jobs list
```

### Manual Queue Management

If you need to manage queues manually:

```bash
# List queues
aws sqs list-queues --queue-name-prefix bec_ --region us-east-1

# Get queue attributes
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks \
  --attribute-names All

# Delete queue (careful!)
aws sqs delete-queue --queue-url https://sqs.us-east-1.amazonaws.com/.../bec_ldv1_tasks
```

## Summary

BEC Orchestration provides a simple CLI for large-scale task processing:

- **Job Management**: Create jobs with automatic queue creation
- **Queue Convention**: `bec_{job_name}_tasks` / `bec_{job_name}_dlq`
- **Automatic Setup**: Queues created when you create a job
- **Simple Usage**: Just use `--job-name` - queue URLs fetched automatically
- **Worker Runtime**: Polls SQS, processes tasks, tracks metrics
- **Monitoring**: CLI commands to view queue stats and job status

For more details, see:
- `QUEUE_CONVENTIONS_SIMPLE.md` - Queue configuration guide
- `FINAL_QUEUE_DESIGN.md` - Design rationale
- `IMPLEMENTATION_SUMMARY.md` - Overall architecture
