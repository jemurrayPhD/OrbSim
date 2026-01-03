from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from orbsim.chem.elements import SYMBOL_TO_ELEMENT
from orbsim.chem.formula_parser import parse_formula


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


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
                    names.append(name)
        else:
            handle.seek(0)
            reader_plain = csv.reader(handle)
            for row in reader_plain:
                if row:
                    name = row[0].strip()
                    if name and name.lower() != "name":
                        names.append(name)
        return names


def build_db(seed_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    create_schema(connection)

    names = load_seed_names(seed_path)
    for idx, name in enumerate(names, start=1):
        print(f"[{idx}/{len(names)}] Resolving {name}…", file=sys.stderr)
        cid = resolve_cid(name)
        if not cid:
            print(f"Skipping {name}: no CID found", file=sys.stderr)
            continue
        props = fetch_properties(cid)
        if not props:
            print(f"Skipping {name}: no properties", file=sys.stderr)
            continue
        formula = props.get("MolecularFormula")
        if not formula:
            print(f"Skipping {name}: no formula", file=sys.stderr)
            continue
        try:
            formula_counts = parse_formula(formula)
        except ValueError:
            print(f"Skipping {name}: formula parse failed ({formula})", file=sys.stderr)
            continue
        element_counts = []
        for symbol, count in formula_counts.items():
            element = SYMBOL_TO_ELEMENT.get(symbol)
            if not element:
                continue
            element_counts.append((element.atomic_number, count))

        data_json = json.dumps({"iupac_name": props.get("IUPACName")})
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
                1,
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
        time.sleep(0.2)
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compound seed database from PubChem.")
    parser.add_argument("--seed", type=Path, default=Path("tools/seed_compounds.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.seed.exists():
        raise SystemExit(f"Seed file not found: {args.seed}")
    build_db(args.seed, args.out)


if __name__ == "__main__":
    main()
