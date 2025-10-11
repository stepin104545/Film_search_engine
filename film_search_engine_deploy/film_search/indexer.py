from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import load_movies, build_canonical_text, project_fields_for_display


INDEX_DIR = "index"


@dataclass
class IndexBundle:
    vectorizer: TfidfVectorizer
    matrix: Any  # sparse matrix
    meta: pd.DataFrame  # id, title, year, genres, director, actors, rating, votes, popularity


def _apply_field_weights(texts: pd.Series, df: pd.DataFrame) -> List[str]:
    # Simple weighting by repeating tokens from certain fields
    def repeat_tokens(s: str, times: int) -> str:
        if not s:
            return ""
        return (" " + s) * max(1, times)

    weighted: List[str] = []
    for i, base in enumerate(texts):
        t = base
        # Title and genres get extra influence; director/actors too
        t += repeat_tokens(str(df.iloc[i].get("title", "")), 2)
        t += repeat_tokens(str(df.iloc[i].get("genres", "")), 2)
        t += repeat_tokens(str(df.iloc[i].get("director", "")), 2)
        t += repeat_tokens(str(df.iloc[i].get("actors", "")), 1)
        weighted.append(t)
    return weighted


def build_index(dataset_base: Optional[Path] = None, out_dir: Optional[Path] = None) -> IndexBundle:
    df = load_movies(dataset_base)
    display_df = project_fields_for_display(df)

    texts = build_canonical_text(df)
    weighted_texts = _apply_field_weights(texts, df)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=200000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(weighted_texts)
    matrix = normalize(matrix)

    bundle = IndexBundle(vectorizer=vectorizer, matrix=matrix, meta=display_df)

    out = Path(out_dir) if out_dir else Path(INDEX_DIR)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "index.joblib")

    return bundle


def load_index(index_dir: Optional[Path] = None) -> IndexBundle:
    base = Path(index_dir) if index_dir else Path(INDEX_DIR)
    path = base / "index.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run the build step first.")
    return joblib.load(path)
