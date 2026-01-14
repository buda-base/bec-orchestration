# BEC Orchestration

This repository contains code to orchestrate tasks (typically OCR) on a large scale at a large scale (60,000 volumes, 22M images, 8TB).

The code is meant to run on EC2, started by systemd, with some cloudwatch logs.

The mental model is:
- a job is a type of task that operates at the volume level (ex: OCR). Jobs can depend on other jobs, like OCR evaluation depends on OCR being done
- a task is the application of a job on one volume. Tasks are typically registered in SQS, two queues per job (one for DLQ)
- the main process running on an EC2 instance is the BECWorker, which (in a loop) reads a task from SQS, runs the appropriate worker for the job on the volume, registers metrics in SQL 
- the job workers typically read images (and / or previous job's artefacts) from S3 and write artefacts on S3

A thin orchestration layer allows creating jobs, queuing tasks and getting metrics.

```
bec_orch/
  README.md
  pyproject.toml
  requirements.txt
  schema.sql             # schema for the POSTGRESQL db
  bec-worker.service     # systemd config for running on EC2 instance

  bec_orch/
    __init__.py

    config.py            # dataclasses + env loading
    logging_setup.py     # structured logging for CloudWatch
    errors.py            # custom exceptions (RetryableError, TerminalError)

    core/                # main entry point for what runs in a worker instance
      __init__.py
      worker_runtime.py  # BECWorker (poll SQS, claim DB, run job worker, write success.json)
      registry.py        # map job_name -> JobWorker class
      models.py          # VolumeRef, TaskMessage, TaskResult, etc.
      metrics.py         # local metrics aggregation (optional)

    io/                  # helpers for DBs
      __init__.py
      db.py              # DBClient: raw SQL, transactions
      sqs.py             # SQSClient wrapper

    jobs/                # job workers
      __init__.py
      base.py            # JobWorker Protocol / ABC + shared helpers for job workers
      shared/            # shared code used across multiple jobs
      ldv1/              # line detection v1 job worker ()

    orch/                # for orchestration of workers (not used on the instance)
      __init__.py
      job_admin.py       # create job, update config, mark job status
      enqueue.py         # enqueue volumes -> SQS
      scaling.py         # spawn/stop workers (later: ASG/ECS helpers)
      discovery.py       # list jobs, list volumes, inspect queue, etc.

    cli/                 # cli wrapper for the orch
      __init__.py
      main.py            # entrypoint: `python -m bec_orch.cli.main`
      worker.py          # `bec worker --job-id ...` runs BECWorker loop
      jobs.py            # `bec jobs create|show|update`
      queue.py           # `bec queue enqueue|stats|redrive`
      scale.py           # `bec scale up|down` (later)
```

Expected env variables:

```
BEC_DEST_S3_BUCKET="tests-bec.bdrc.io"
BEC_SQL_HOST="..."
BEC_SQL_USER="..."
BEC_SQL_PORT="..."
BEC_SQL_PASSWORD="..."
BEC_REGION="us-east-1"
```