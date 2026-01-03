from __future__ import annotations

import argparse
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
PUBCHEM_CITATION_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines"

SEED_COMPOUNDS = [
    "water",
    "hydrogen peroxide",
    "carbon dioxide",
    "carbon monoxide",
    "methane",
    "ethane",
    "propane",
    "butane",
    "ethanol",
    "methanol",
    "glucose",
    "sucrose",
    "sodium chloride",
    "potassium chloride",
    "ammonia",
    "nitric acid",
    "sulfuric acid",
    "hydrochloric acid",
    "phosphoric acid",
    "calcium carbonate",
    "sodium bicarbonate",
    "calcium hydroxide",
    "sodium hydroxide",
    "magnesium sulfate",
    "copper sulfate",
    "iron(iii) oxide",
    "iron(ii) oxide",
    "aluminum oxide",
    "silicon dioxide",
    "sodium nitrate",
    "potassium nitrate",
    "sodium sulfate",
    "ammonium chloride",
    "ammonium nitrate",
    "acetic acid",
    "formic acid",
    "citric acid",
    "urea",
    "benzene",
    "toluene",
    "phenol",
    "acetone",
    "formaldehyde",
    "glycine",
    "alanine",
    "sodium hypochlorite",
    "calcium chloride",
    "magnesium chloride",
    "sodium fluoride",
    "sodium bromide",
    "potassium bromide",
    "ammonium sulfate",
    "sulfur dioxide",
    "nitrogen dioxide",
    "nitrous oxide",
    "ozone",
    "chlorine",
    "bromine",
]


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


def fetch_synonyms(cid: int) -> list[str]:
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    try:
        data = fetch_json(url)
    except Exception:
        return []
    return data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])[:10]


def create_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS compounds;
        DROP TABLE IF EXISTS compound_elements;
        DROP TABLE IF EXISTS citations;
        CREATE TABLE compounds (
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
        CREATE TABLE compound_elements (
            cid INTEGER,
            atomic_number INTEGER,
            count INTEGER,
            PRIMARY KEY (cid, atomic_number)
        );
        CREATE TABLE citations (
            source_id TEXT PRIMARY KEY,
            label TEXT,
            url TEXT
        );
        """
    )
    connection.commit()


def insert_citations(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO citations (source_id, label, url) VALUES (?, ?, ?)",
        ("pubchem", "PubChem (NIH/NLM) citation guidelines", PUBCHEM_CITATION_URL),
    )
    connection.commit()


def build_db(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    create_schema(connection)
    insert_citations(connection)

    for name in SEED_COMPOUNDS:
        cid = resolve_cid(name)
        if not cid:
            print(f"Skipping {name}: no CID found")
            continue
        props = fetch_properties(cid)
        if not props:
            print(f"Skipping {name}: no properties")
            continue
        formula = props.get("MolecularFormula")
        if not formula:
            print(f"Skipping {name}: no formula")
            continue
        try:
            formula_counts = parse_formula(formula)
        except ValueError:
            print(f"Skipping {name}: formula parse failed ({formula})")
            continue
        element_counts = []
        for symbol, count in formula_counts.items():
            element = SYMBOL_TO_ELEMENT.get(symbol)
            if not element:
                continue
            element_counts.append((element.atomic_number, count))

        data_json = json.dumps(
            {
                "iupac_name": props.get("IUPACName"),
                "synonyms": fetch_synonyms(cid),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "orbsim" / "data" / "compounds.sqlite",
    )
    args = parser.parse_args()
    build_db(args.output)


if __name__ == "__main__":
    main()
