from __future__ import annotations

import json as _json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()  # loads .env; system env vars override .env values

BASE_DIR = Path(__file__).resolve().parents[1]
_data_override = os.environ.get("WEB_DATA_DIR", "").strip()
DATA_DIR = Path(_data_override) if _data_override else BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

INITDB_SQLITE = Path(__file__).parent / "initdb_sqlite.sql"
INITDB_MYSQL = Path(__file__).parent / "initdb_mysql.sql"

DEPLOY_TYPE = os.environ.get("DEPLOY_TYPE", "DEV").strip().upper()

if DEPLOY_TYPE == "PROD":
    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ.get("MYSQL_USER", "internta_user"),
        password=os.environ["MYSQL_PASSWORD"],
        host=os.environ.get("MYSQL_HOST", "mysql6.sqlpub.com"),
        port=int(os.environ.get("MYSQL_PORT", "3311")),
        database=os.environ.get("MYSQL_DATABASE", "internta_db"),
    )
else:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = URL.create(
        drivername="sqlite+pysqlite",
        database=str(DB_PATH),
    )

engine = create_engine(url)


def init_db() -> None:
    sql_file = INITDB_MYSQL if DEPLOY_TYPE == "PROD" else INITDB_SQLITE
    sql = Path(sql_file).read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _last_insert_id(result) -> int:
    return int(result.lastrowid)


def insert_note(content: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO notes (content) VALUES (:content)"),
            {"content": content},
        )
        return _last_insert_id(result)


def list_notes() -> list:
    with engine.connect() as conn:
        return list(
            conn.execute(
                text("SELECT id, content, created_at FROM notes ORDER BY id DESC")
            ).mappings().fetchall()
        )


def get_note(note_id: int) -> Optional[object]:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT id, content, created_at FROM notes WHERE id = :id"),
            {"id": note_id},
        ).mappings().fetchone()


def insert_action_items(items: list[str], note_id: Optional[int] = None) -> list[int]:
    ids: list[int] = []
    with engine.begin() as conn:
        for item in items:
            result = conn.execute(
                text("INSERT INTO action_items (note_id, text) VALUES (:note_id, :text)"),
                {"note_id": note_id, "text": item},
            )
            ids.append(_last_insert_id(result))
    return ids


def list_action_items(note_id: Optional[int] = None) -> list:
    with engine.connect() as conn:
        if note_id is None:
            rows = conn.execute(
                text(
                    "SELECT id, note_id, text, done, created_at"
                    " FROM action_items ORDER BY id DESC"
                )
            ).mappings().fetchall()
        else:
            rows = conn.execute(
                text(
                    "SELECT id, note_id, text, done, created_at"
                    " FROM action_items WHERE note_id = :note_id ORDER BY id DESC"
                ),
                {"note_id": note_id},
            ).mappings().fetchall()
        return list(rows)


def mark_action_item_done(action_item_id: int, done: bool) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE action_items SET done = :done WHERE id = :id"),
            {"done": 1 if done else 0, "id": action_item_id},
        )


def insert_opm_diagram(payload: dict, note_id: Optional[int] = None) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO opm_diagrams (note_id, payload) VALUES (:note_id, :payload)"
            ),
            {"note_id": note_id, "payload": _json.dumps(payload)},
        )
        return _last_insert_id(result)


def list_opm_diagrams(limit: Optional[int] = None) -> list[dict]:
    q = "SELECT id, note_id, payload, created_at FROM opm_diagrams ORDER BY id DESC"
    params: dict = {}
    if limit is not None:
        q += " LIMIT :limit"
        params["limit"] = int(limit)
    with engine.connect() as conn:
        rows = conn.execute(text(q), params).fetchall()
    return [
        {
            "id": r.id,
            "note_id": r.note_id,
            "created_at": r.created_at,
            "diagram": _json.loads(r.payload),
        }
        for r in rows
    ]


def get_opm_diagram(diagram_id: int) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, note_id, payload, created_at"
                " FROM opm_diagrams WHERE id = :id"
            ),
            {"id": diagram_id},
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row.id,
        "note_id": row.note_id,
        "created_at": row.created_at,
        "diagram": _json.loads(row.payload),
    }
