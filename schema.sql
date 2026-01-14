-- Assumes: CREATE DATABASE pipeline_v1 ENCODING 'UTF8';

-- Types ------------------------------------------------------------------

CREATE TYPE job_status AS ENUM (
  'created',
  'running',
  'completed',
  'failed',
  'canceled'
);

CREATE TYPE task_status AS ENUM (
  'running',
  'done',
  'retryable_failed',
  'terminal_failed'
);

-- Tables -----------------------------------------------------------------

CREATE TABLE jobs (
  id                   serial PRIMARY KEY,
  name                 varchar(50) NOT NULL UNIQUE,
  config               text, -- most of the time json but the field is free
  queue_url            text NOT NULL, -- SQS queue URL for tasks
  dlq_url              text, -- optional dead-letter queue URL
  created_at           timestamptz NOT NULL DEFAULT now(),
  updated_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE volumes (
  id        serial PRIMARY KEY,
  bdrc_w_id varchar(32) NOT NULL,
  bdrc_i_id varchar(32) NOT NULL,
  -- the last modification of the dimensions.json reported on s3
  last_s3_etag bytea  NOT NULL CHECK (octet_length(last_s3_etag) = 16), -- the s3 etag of the manifest
  last_modified_at timestamptz, -- from s3
  -- number of images in the volume, mostly informational, shouldn't be relied on
  nb_images smallint NOT NULL,
  -- number of "intro images", images to ignore at the beginning of the volumes, for all jobs
  nb_images_intro smallint NOT NULL
);

CREATE UNIQUE INDEX volumes_w_i_uniq ON volumes (bdrc_w_id, bdrc_i_id);
CREATE INDEX volumes_bdrc_w_id_idx ON volumes (bdrc_w_id);

CREATE TABLE workers (
  worker_id         serial PRIMARY KEY,
  worker_name       text, -- by construction ec2 instance id
  hostname          text,
  tags              jsonb,
  started_at        timestamptz NOT NULL DEFAULT now(),
  last_heartbeat_at timestamptz NOT NULL DEFAULT now(),
  stopped_at        timestamptz
);

CREATE INDEX workers_last_heartbeat_idx ON workers (last_heartbeat_at);
CREATE INDEX workers_worker_name_idx ON workers (worker_name);

CREATE TABLE task_executions (
  id                  bigserial PRIMARY KEY,
  job_id              bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  volume_id           integer NOT NULL REFERENCES volumes(id),
  s3_etag             bytea  NOT NULL CHECK (octet_length(s3_etag) = 16), -- the s3 etag of the manifest
  status              task_status NOT NULL,
  attempt             integer NOT NULL DEFAULT 0,
  worker_id           integer REFERENCES workers(worker_id),
  started_at          timestamptz,
  done_at             timestamptz,
  total_images        smallint,
  nb_errors           smallint,
  total_duration_ms        real,
  avg_duration_per_page_ms real
);

CREATE UNIQUE INDEX task_executions_job_volume_etag_uniq ON task_executions (job_id, volume_id, s3_etag);
CREATE INDEX tasks_job_id_idx ON task_executions (job_id);
CREATE INDEX tasks_job_status_idx ON task_executions (job_id, status);
CREATE INDEX tasks_status_idx ON task_executions (status);
CREATE INDEX tasks_volume_id_idx ON task_executions (volume_id);
