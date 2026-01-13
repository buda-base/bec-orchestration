# BEC Orchestration

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

See 