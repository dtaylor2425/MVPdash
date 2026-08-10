from __future__ import annotations

import os
from typing import Optional

import psycopg
from psycopg.rows import dict_row


def get_database_url() -> str:
    database_url: Optional[str] = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Add Railway Postgres and reference its DATABASE_URL.")
    return database_url


def get_connection() -> psycopg.Connection:
    return psycopg.connect(get_database_url(), row_factory=dict_row, autocommit=False)
