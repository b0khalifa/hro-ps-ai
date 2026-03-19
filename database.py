import os

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "hro_db"

DEFAULT_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Keep this module import-safe (no DB calls on import).
_raw_database_url = os.getenv("DATABASE_URL")
DATABASE_URL = (
    (_raw_database_url.strip() if isinstance(_raw_database_url, str) else "")
    or DEFAULT_DATABASE_URL
).rstrip("/")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    # Production-friendly defaults while keeping local-dev simple.
    # - pool_pre_ping avoids stale connections
    # - pool_recycle helps with network timeouts
    return create_engine(
        DATABASE_URL,
        echo=_env_bool("SQLALCHEMY_ECHO", False),
        pool_pre_ping=True,
        pool_recycle=_env_int("SQLALCHEMY_POOL_RECYCLE", 1800),
        pool_size=_env_int("SQLALCHEMY_POOL_SIZE", 5),
        max_overflow=_env_int("SQLALCHEMY_MAX_OVERFLOW", 10),
    )


# Backwards-compatible exports (other modules import these symbols)
engine: Engine = get_engine()
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)
Base = declarative_base()


def init_db() -> None:
    """Create tables for all imported models.

    In production prefer migrations (Alembic). In this repo we keep create_all as
    a dev-friendly default.
    """

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy Session per request."""

    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope(commit: bool = False) -> Iterator[Session]:
    """Reusable session scope for scripts/Streamlit helpers.

    Args:
        commit: when True, commits on success and rolls back on exception.
    """

    db: Session = SessionLocal()
    try:
        yield db
        if commit:
            db.commit()
    except Exception:
        if commit:
            db.rollback()
        raise
    finally:
        db.close()
