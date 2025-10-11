import gzip
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


MOVIES_CSV = "msrd-dataset/msrd/dataset/movies.csv.gz"
MOVIES_JSONL = "msrd-dataset/msrd/dataset/movies.jsonl.gz"


def _read_movies_csv(path: Path) -> pd.DataFrame:
    # Use python engine with quotechar to handle embedded newlines properly
    df = pd.read_csv(
        path, 
        sep="\t", 
        quotechar='"',
        engine='python',
        on_bad_lines='skip',
        encoding='utf-8'
    )
    # Ensure expected columns exist; fill missing
    expected = [
        "id",
        "title",
        "overview",
        "tags",
        "genres",
        "director",
        "actors",
        "characters",
        "year",
        "votes",
        "rating",
        "popularity",
        "budget",
        "poster_url",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    # Normalize types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["votes", "rating", "popularity", "budget"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["overview", "tags", "genres", "director", "actors", "characters"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    df["title"] = df["title"].fillna("")
    return df


def _read_movies_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if not df.empty and "id" in df.columns:
        df.drop_duplicates(subset=["id"], inplace=True)
    return df


def load_movies(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load MSRD movies metadata as a DataFrame.

    Prefers the TSV CSV file (authoritative). Falls back to JSONL if present.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    csv_path = (base / MOVIES_CSV).resolve()
    jsonl_path = (base / MOVIES_JSONL).resolve()

    if csv_path.exists():
        return _read_movies_csv(csv_path)
    elif jsonl_path.exists():
        return _read_movies_jsonl(jsonl_path)
    else:
        raise FileNotFoundError(
            f"Could not find movies data. Looked for {csv_path} and {jsonl_path}."
        )


def normalize_list_field(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]


def build_canonical_text(df: pd.DataFrame) -> pd.Series:
    """
    Create a single text field per movie with weighted concatenation guidance left to the indexer.
    Here we just provide fields separately in a dict-like string to keep simple TF-IDF usage possible.
    """
    parts: List[str] = []
    titles = df["title"].fillna("")
    overviews = df["overview"].fillna("")
    tags = df["tags"].fillna("")
    genres = df["genres"].fillna("")
    director = df["director"].fillna("")
    actors = df["actors"].fillna("")
    characters = df["characters"].fillna("")

    # Simple concatenation; weights applied later by repeating tokens
    combined = (
        titles.str.cat(overviews, sep=". ")
        .str.cat(genres, sep=". ")
        .str.cat(tags, sep=", ")
        .str.cat(director, sep=", ")
        .str.cat(actors, sep=", ")
        .str.cat(characters, sep=", ")
    )
    return combined.fillna("")


def project_fields_for_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["id", "title", "year", "genres", "director", "actors", "rating", "votes", "popularity"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()
