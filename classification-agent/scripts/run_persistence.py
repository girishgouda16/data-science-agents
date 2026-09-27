"""Best-effort write-through to the shared `training_runs` table (schema
owned by the repo-root Alembic migrations, gateway/core/db.py has the same
table for its own writers). Deliberately NOT an import of core.db — this
agent stays independently deployable (own requirements.txt, no cross-agent
imports, per README) — so the table is duplicated here as plain SQLAlchemy
Core, same shape as core/db.py's `training_runs`, small enough that
duplication is cheaper than a shared dependency.

DATABASE_URL env var picks the backend; defaults to the same SQLite file
core/config.py defaults to (resolved as an absolute path so it's the same
file regardless of which directory the agent process is launched from).
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
_default_db = f"sqlite:///{REPO_ROOT / 'data' / 'training_runs.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL", _default_db)

_metadata = MetaData()
_training_runs = Table(
    "training_runs",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("task_id", String, nullable=False),
    Column("target_column", String, nullable=False),
    Column("best_model", String, nullable=False),
    Column("metric", String, nullable=False),
    Column("best_score", Float, nullable=False),
    Column("test_auc", Float),
    Column("cv_to_test_gap", Float),
    Column("overfitting_warning", Boolean),
    Column("all_model_scores", JSON),
    Column("best_params", JSON),
    Column("feature_importance", JSON),
    Column("label_classes", JSON),
    Column("artifact_path", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def save_training_run(
    run_id: str,
    *,
    target_column: str,
    best_model: str,
    metric: str,
    best_score: float,
    test_auc: float | None = None,
    cv_to_test_gap: float | None = None,
    overfitting_warning: bool | None = None,
    all_model_scores: Any = None,
    best_params: Any = None,
    feature_importance: Any = None,
    label_classes: Any = None,
    artifact_path: str | None = None,
) -> None:
    """Insert one row per train_model/tune_hyperparams call. run_id doubles
    as both session_id and task_id — this agent has no separate A2A
    session concept, run_id is its one persistent handle (see mcp_server.py).
    Never raises: a DB hiccup must not fail a training run (same contract
    as core/db.py's save_training_run)."""
    try:
        with _get_engine().begin() as conn:
            conn.execute(
                _training_runs.insert().values(
                    session_id=run_id,
                    task_id=run_id,
                    target_column=target_column,
                    best_model=best_model,
                    metric=metric,
                    best_score=best_score,
                    test_auc=test_auc,
                    cv_to_test_gap=cv_to_test_gap,
                    overfitting_warning=overfitting_warning,
                    all_model_scores=all_model_scores,
                    best_params=best_params,
                    feature_importance=feature_importance,
                    label_classes=label_classes,
                    artifact_path=artifact_path,
                    created_at=datetime.now(timezone.utc),
                )
            )
    except Exception:
        logger.exception(
            "Could not persist training_runs row for run_id=%s (continuing)", run_id
        )
