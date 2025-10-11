from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz, process
import pandas as pd

from .indexer import load_index, IndexBundle
from .query import ParsedQuery


@dataclass
class SearchResult:
    idx: int
    movie_id: int
    title: str
    year: Optional[int]
    genres: str
    director: str
    actors: str
    score: float


def _filter_indices(q: ParsedQuery, bundle: IndexBundle) -> List[int]:
    meta = bundle.meta
    mask = np.ones(len(meta), dtype=bool)

    if q.year_from is not None:
        y = meta.get("year")
        if y is not None:
            yvals = y.fillna(0).astype(int).values
            mask &= yvals >= q.year_from
    if q.year_to is not None:
        y = meta.get("year")
        if y is not None:
            yvals = y.fillna(9999).astype(int).values
            mask &= yvals <= q.year_to

    if q.genres:
        gcol = meta.get("genres")
        if gcol is not None:
            mm = np.zeros(len(meta), dtype=bool)
            gvals = gcol.fillna("").str.lower().values
            for g in q.genres:
                mm |= np.array([g in s for s in gvals])
            mask &= mm

    if q.people:
        dcol = meta.get("director")
        acol = meta.get("actors")
        mm = np.zeros(len(meta), dtype=bool)
        dvals = dcol.fillna("").values if dcol is not None else [""] * len(meta)
        avals = acol.fillna("").values if acol is not None else [""] * len(meta)
        for name in q.people:
            low = name.lower()
            mm |= np.array([low in str(d).lower() or low in str(a).lower() for d, a in zip(dvals, avals)])
        mask &= mm

    return np.where(mask)[0].tolist()


def _text_score(q: ParsedQuery, bundle: IndexBundle, candidate_idx: List[int]) -> np.ndarray:
    query_text = q.text if q.text else q.raw
    vec = bundle.vectorizer.transform([query_text])
    vec = vec / (np.sqrt((vec.multiply(vec)).sum()) + 1e-9)
    sub = bundle.matrix[candidate_idx]
    sims = sub.dot(vec.T).toarray().ravel()
    return sims


def _fuzzy_boost(q: ParsedQuery, bundle: IndexBundle, candidate_idx: List[int]) -> np.ndarray:
    titles = bundle.meta.iloc[candidate_idx]["title"].fillna("").tolist()
    if not q.raw.strip():
        return np.zeros(len(candidate_idx))
    boost = np.zeros(len(candidate_idx))
    for i, t in enumerate(titles):
        boost[i] = fuzz.partial_ratio(q.raw.lower(), t.lower()) / 100.0 * 0.5
    return boost


def _metadata_boost(q: ParsedQuery, bundle: IndexBundle, candidate_idx: List[int]) -> np.ndarray:
    """Boost scores based on metadata matches"""
    boost = np.zeros(len(candidate_idx))
    meta_subset = bundle.meta.iloc[candidate_idx]
    
    # Boost for genre matches
    if q.genres:
        for i, idx in enumerate(candidate_idx):
            genres_str = str(meta_subset.iloc[i].get("genres", "")).lower()
            for genre in q.genres:
                if genre in genres_str:
                    boost[i] += 0.3
    
    # Boost for people matches (actor/director)
    if q.people:
        for i, idx in enumerate(candidate_idx):
            actors_str = str(meta_subset.iloc[i].get("actors", "")).lower()
            director_str = str(meta_subset.iloc[i].get("director", "")).lower()
            for person in q.people:
                person_lower = person.lower()
                if person_lower in actors_str or person_lower in director_str:
                    boost[i] += 0.4
    
    # Boost for year exact match
    if q.year_from is not None and q.year_to is not None:
        for i, idx in enumerate(candidate_idx):
            year = meta_subset.iloc[i].get("year")
            if not pd.isna(year):
                year_int = int(year)
                # Boost movies in the middle of the range
                range_size = q.year_to - q.year_from + 1
                if range_size > 1:
                    mid_year = (q.year_from + q.year_to) / 2
                    distance = abs(year_int - mid_year)
                    boost[i] += 0.2 * (1 - distance / (range_size / 2))
                else:
                    boost[i] += 0.2
    
    # Boost for popularity/rating
    for i, idx in enumerate(candidate_idx):
        rating = meta_subset.iloc[i].get("rating")
        votes = meta_subset.iloc[i].get("votes")
        if not pd.isna(rating) and not pd.isna(votes):
            if float(rating) >= 7.0 and int(votes) >= 1000:
                boost[i] += 0.15
    
    return boost


def search(q: ParsedQuery, top_k: int = 20, index_dir: Optional[str] = None) -> List[SearchResult]:
    bundle = load_index(index_dir)
    idxs = _filter_indices(q, bundle)
    if len(idxs) == 0:
        return []

    # Combine multiple scoring signals
    text_scores = _text_score(q, bundle, idxs)
    fuzzy_scores = _fuzzy_boost(q, bundle, idxs)
    metadata_scores = _metadata_boost(q, bundle, idxs)
    
    # Weighted combination
    sims = (text_scores * 0.4) + (fuzzy_scores * 0.2) + (metadata_scores * 0.4)
    
    # Normalize to 0-1 range
    if len(sims) > 0 and sims.max() > 0:
        sims = sims / sims.max()

    order = np.argsort(-sims)[:top_k]
    res: List[SearchResult] = []
    for rank in order:
        i = idxs[rank]
        row = bundle.meta.iloc[i]
        res.append(
            SearchResult(
                idx=i,
                movie_id=int(row.get("id", -1)),
                title=str(row.get("title", "")),
                year=int(row.get("year")) if not pd.isna(row.get("year")) else None,  # type: ignore
                genres=str(row.get("genres", "")),
                director=str(row.get("director", "")),
                actors=str(row.get("actors", "")),
                score=float(sims[rank]),
            )
        )
    return res
