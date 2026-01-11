from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orbsim.chem.elements import get_atomic_number
from orbsim.chem.formula_parser import parse_formula


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_LIMIT_S = 0.25
MAX_RETRIES = 3
RETRY_BACKOFF_S = 0.5
_last_request_ts = 0.0


@dataclass
class BuildReporter:
    log: callable
    progress: callable

    def info(self, message: str) -> None:
        if self.log:
            self.log(message)

    def update(self, current: int, total: int, message: str | None = None) -> None:
        if self.progress:
            self.progress(current, total)
        if message:
            self.info(message)


def _rate_limit() -> None:
    global _last_request_ts
    now = time.monotonic()
    delta = now - _last_request_ts
    if delta < RATE_LIMIT_S:
        time.sleep(RATE_LIMIT_S - delta)
    _last_request_ts = time.monotonic()


def fetch_json(url: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(RETRY_BACKOFF_S * (2**attempt))
    if last_error:
        raise last_error
    return {}


def resolve_cid(name: str) -> int | None:
    encoded = urllib.parse.quote(name)
    url = f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
    try:
        data = fetch_json(url)
    except Exception:
        return None
    cids = data.get("IdentifierList", {}).get("CID", [])
    return int(cids[0]) if cids else None


def fetch_properties(cid: int) -> dict | None:
    url = (
        f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
        "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChIKey,Title/JSON"
    )
    try:
        data = fetch_json(url)
    except Exception:
        return None
    props = data.get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else None


def fetch_synonyms(cid: int) -> list[str]:
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    try:
        data = fetch_json(url)
    except Exception:
        return []
    info = data.get("InformationList", {}).get("Information", [])
    if not info:
        return []
    return info[0].get("Synonym", []) or []


def create_schema(connection: sqlite3.Connection) -> None:
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
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    connection.commit()


def load_seed_names(seed_path: Path) -> list[str]:
    with seed_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = []
        if reader.fieldnames and "name" in reader.fieldnames:
            for row in reader:
                name = (row.get("name") or "").strip()
                if name:
                        for part in name.split(";"):
                            entry = part.strip()
                            if entry:
                                names.append(entry)
        else:
            handle.seek(0)
            reader_plain = csv.reader(handle)
            for row in reader_plain:
                if row:
                    name = row[0].strip()
                    if name and name.lower() != "name":
                        for part in name.split(";"):
                            entry = part.strip()
                            if entry:
                                names.append(entry)
        return names


def _existing_cids(connection: sqlite3.Connection) -> set[int]:
    rows = connection.execute("SELECT cid FROM compounds").fetchall()
    return {int(row[0]) for row in rows}


def _update_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (key, value),
    )
    connection.commit()


def build_db(
    seed_path: Path,
    output: Path,
    mode: str = "rebuild",
    limit: int | None = None,
    reporter: BuildReporter | None = None,
    cancel_event: object | None = None,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    create_schema(connection)

    if mode == "rebuild":
        connection.execute("DELETE FROM compound_elements")
        connection.execute("DELETE FROM compounds")
        connection.commit()
        existing = set()
    else:
        existing = _existing_cids(connection)

    names = load_seed_names(seed_path)
    if limit:
        names = names[:limit]
    total = len(names)
    cancelled = False
    for idx, name in enumerate(names, start=1):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            if reporter:
                reporter.info("Cancelled.")
            cancelled = True
            break
        if reporter:
            reporter.update(idx, total, f"Resolving CID {idx}/{total}: {name}")
        else:
            print(f"[{idx}/{len(names)}] Resolving {name}…", file=sys.stderr)
        try:
            cid = resolve_cid(name)
        except Exception as exc:
            if reporter:
                reporter.info(f"Error resolving {name}: {exc}")
            else:
                print(f"Error resolving {name}: {exc}", file=sys.stderr)
            continue
        if not cid:
            if reporter:
                reporter.info(f"Skipping {name}: no CID found")
            else:
                print(f"Skipping {name}: no CID found", file=sys.stderr)
            continue
        if cid in existing:
            if reporter:
                reporter.info(f"Skipping {name}: CID already in database")
            continue
        if reporter:
            reporter.info("Fetching properties…")
        try:
            props = fetch_properties(cid)
        except Exception as exc:
            if reporter:
                reporter.info(f"Error fetching properties for {name}: {exc}")
            else:
                print(f"Error fetching properties for {name}: {exc}", file=sys.stderr)
            continue
        if not props:
            if reporter:
                reporter.info(f"Skipping {name}: no properties")
            else:
                print(f"Skipping {name}: no properties", file=sys.stderr)
            continue
        try:
            synonyms = fetch_synonyms(cid)
        except Exception as exc:
            synonyms = []
            if reporter:
                reporter.info(f"Synonym fetch failed for {name}: {exc}")
        formula = props.get("MolecularFormula")
        if not formula:
            if reporter:
                reporter.info(f"Skipping {name}: no formula")
            else:
                print(f"Skipping {name}: no formula", file=sys.stderr)
            continue
        try:
            formula_counts = parse_formula(formula)
        except ValueError:
            if reporter:
                reporter.info(f"Skipping {name}: formula parse failed ({formula})")
            else:
                print(f"Skipping {name}: formula parse failed ({formula})", file=sys.stderr)
            continue
        element_counts = []
        for symbol, count in formula_counts.items():
            atomic_number = get_atomic_number(symbol)
            if not atomic_number:
                continue
            element_counts.append((atomic_number, count))

        data_json = json.dumps(
            {
                "iupac_name": props.get("IUPACName"),
                "title": props.get("Title") or name.title(),
                "synonyms": synonyms,
            }
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO compounds
            (cid, name, formula, smiles, inchikey, mol_weight, is_seed, pubchem_url, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                props.get("Title") or name.title(),
                formula,
                props.get("CanonicalSMILES"),
                props.get("InChIKey"),
                props.get("MolecularWeight"),
                1 if mode == "rebuild" else 0,
                f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                data_json,
            ),
        )
        for atomic_number, count in element_counts:
            connection.execute(
                "INSERT OR REPLACE INTO compound_elements (cid, atomic_number, count) VALUES (?, ?, ?)",
                (cid, atomic_number, count),
            )
        connection.commit()
        existing.add(cid)
    if not cancelled:
        _update_metadata(connection, "last_built", time.strftime("%Y-%m-%d %H:%M:%S"))
    connection.close()
    return not cancelled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compound seed database from PubChem.")
    parser.add_argument("--seed", type=Path, default=Path("tools/seed_compounds.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["rebuild", "append"], default="rebuild")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.seed.exists():
        raise SystemExit(f"Seed file not found: {args.seed}")
    build_db(args.seed, args.out, mode=args.mode, limit=args.limit)


if __name__ == "__main__":
    main()
