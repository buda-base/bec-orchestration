from __future__ import annotations
from typing import Optional
import psycopg

from bec_orch.io.db import DBClient
from bec_orch.core.models import JobRecord

def create_job(conn: psycopg.Connection, name: str, config_text: str) -> int: ...

def get_job(conn: psycopg.Connection, job_id: int) -> JobRecord: ...
