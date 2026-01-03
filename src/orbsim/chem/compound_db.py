from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path


DB_FILENAME = "compounds.sqlite"


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / DB_FILENAME


def load_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS compounds (
            cid INTEGER PRIMARY KEY,
            name TEXT,
            formula TEXT,
            smiles TEXT,
            inchikey TEXT,
            mol_weight REAL,
            is_seed INTEGER,
            pubchem_url TEXT,
            data_json TEXT
        );
        CREATE TABLE IF NOT EXISTS compound_elements (
            cid INTEGER,
            atomic_number INTEGER,
            count INTEGER,
            PRIMARY KEY (cid, atomic_number)
        );
        CREATE TABLE IF NOT EXISTS citations (
            source_id TEXT PRIMARY KEY,
            label TEXT,
            url TEXT
        );
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO citations (source_id, label, url)
        VALUES (?, ?, ?)
        """,
        (
            "pubchem",
            "PubChem (NIH/NLM) citation guidelines",
            "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines",
        ),
    )
    connection.commit()


def query_compounds_by_elements(
    multiset: list[int],
    formula: str | None = None,
    search: str | None = None,
    limit: int = 200,
) -> list[dict]:
    if not multiset:
        return []
    counts = Counter(multiset)
    element_ids = list(counts.keys())
    placeholders = ",".join("?" for _ in element_ids)
    having_clauses = []
    having_params: list[object] = []
    for atomic_number, count in counts.items():
        having_clauses.append(
            "SUM(CASE WHEN ce.atomic_number = ? AND ce.count >= ? THEN 1 ELSE 0 END) = 1"
        )
        having_params.extend([atomic_number, count])
    having_sql = " AND ".join(having_clauses)
    search_sql = ""
    search_params: list[object] = []
    if search:
        search_sql = "AND (c.name LIKE ? OR c.formula LIKE ?)"
        search_params.extend([f"%{search}%", f"%{search}%"])

    required_total = sum(counts.values())

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
        {search_sql}
        GROUP BY c.cid
        HAVING {having_sql}
        ORDER BY c.is_seed DESC, c.name ASC
        LIMIT {limit}
    """
    params = element_ids + search_params + having_params
    with load_db() as connection:
        rows = connection.execute(query, params).fetchall()
    results = []
    for row in rows:
        extra_atoms = max(int(row["total_atoms"] or 0) - required_total, 0)
        exact_formula = formula is not None and row["formula"] and row["formula"].lower() == formula.lower()
        results.append(
            {
                "cid": row["cid"],
                "name": row["name"],
                "formula": row["formula"],
                "pubchem_url": row["pubchem_url"],
                "is_seed": bool(row["is_seed"]),
                "extra_atoms": extra_atoms,
                "exact_formula": exact_formula,
            }
        )
    results.sort(
        key=lambda item: (
            0 if item["exact_formula"] else 1,
            item["extra_atoms"],
            0 if item["is_seed"] else 1,
            item["name"] or "",
        )
    )
    return results


def get_compound_details(cid: int) -> dict | None:
    with load_db() as connection:
        row = connection.execute(
            "SELECT * FROM compounds WHERE cid = ?",
            (cid,),
        ).fetchone()
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
            "formula": row["formula"],
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "mol_weight": row["mol_weight"],
            "pubchem_url": row["pubchem_url"],
            "elements": {int(elem["atomic_number"]): int(elem["count"]) for elem in elements},
        }
    )
    return data


def get_citations() -> list[dict]:
    with load_db() as connection:
        rows = connection.execute("SELECT source_id, label, url FROM citations ORDER BY source_id").fetchall()
    return [{"source_id": row["source_id"], "label": row["label"], "url": row["url"]} for row in rows]
