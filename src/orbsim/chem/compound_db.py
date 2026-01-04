from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PySide6 import QtCore


DB_FILENAME = "compounds.sqlite"


def get_db_path() -> Path:
    base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
    path = Path(base) / DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def db_exists() -> bool:
    return get_db_path().exists()


def connect() -> sqlite3.Connection:
    path = get_db_path()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def get_compound_count() -> int:
    if not db_exists():
        return 0
    with connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM compounds").fetchone()
    return int(row["count"] if row else 0)


def get_last_built() -> str | None:
    if not db_exists():
        return None
    with connect() as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            ("last_built",),
        ).fetchone()
    return row["value"] if row else None


def query_compounds_by_elements(
    required_counts: dict[int, int],
    limit: int = 200,
    only_elements: bool = False,
) -> list[dict]:
    if not required_counts:
        return []
    element_ids = list(required_counts.keys())
    placeholders = ",".join("?" for _ in element_ids)
    having_clauses = []
    having_params: list[object] = []
    comparator = ">="
    for atomic_number, count in required_counts.items():
        required = 1 if only_elements else count
        having_clauses.append(
            f"SUM(CASE WHEN ce.atomic_number = ? AND ce.count {comparator} ? THEN 1 ELSE 0 END) = 1"
        )
        having_params.extend([atomic_number, required])
    element_filter = ""
    if only_elements:
        element_filter = (
            f"AND NOT EXISTS (SELECT 1 FROM compound_elements ce2 "
            f"WHERE ce2.cid = c.cid AND ce2.atomic_number NOT IN ({placeholders}))"
        )
    having_sql = " AND ".join(having_clauses)

    query = f"""
        SELECT
            c.cid,
            c.name,
            c.formula,
            c.pubchem_url,
            c.is_seed,
            SUM(ce.count) AS total_atoms
        FROM compounds c
        JOIN compound_elements ce ON c.cid = ce.cid
        WHERE ce.atomic_number IN ({placeholders})
        {element_filter}
        GROUP BY c.cid
        HAVING {having_sql}
        ORDER BY c.is_seed DESC, c.name ASC
        LIMIT {limit}
    """
    params = element_ids + (element_ids if only_elements else []) + having_params
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    results = []
    required_total = sum(required_counts.values())
    for row in rows:
        extra_atoms = max(int(row["total_atoms"] or 0) - required_total, 0)
        results.append(
            {
                "cid": row["cid"],
                "name": row["name"],
                "formula": row["formula"],
                "pubchem_url": row["pubchem_url"],
                "is_seed": bool(row["is_seed"]),
                "extra_atoms": extra_atoms,
            }
        )
    results.sort(key=lambda item: (item["extra_atoms"], 0 if item["is_seed"] else 1, item["name"] or ""))
    return results


def get_compound_details(cid: int) -> dict | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM compounds WHERE cid = ?", (cid,)).fetchone()
        if row is None:
            return None
        elements = connection.execute(
            "SELECT atomic_number, count FROM compound_elements WHERE cid = ?",
            (cid,),
        ).fetchall()
    data_json = row["data_json"] or "{}"
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        data = {}
    data.update(
        {
            "cid": row["cid"],
            "name": row["name"],
            "title": data.get("title") or row["name"],
            "formula": row["formula"],
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "mol_weight": row["mol_weight"],
            "pubchem_url": row["pubchem_url"],
            "elements": {int(elem["atomic_number"]): int(elem["count"]) for elem in elements},
        }
    )
    return data
