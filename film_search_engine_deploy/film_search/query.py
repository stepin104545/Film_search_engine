import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


GENRE_SYNONYMS = {
    "sci fi": "science fiction",
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "romcom": "romance,comedy",
    "kid": "family",
    "kids": "family",
}

ALL_GENRES = {
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "history",
    "horror",
    "music",
    "mystery",
    "romance",
    "science fiction",
    "sci-fi",
    "thriller",
    "war",
    "western",
}


@dataclass
class ParsedQuery:
    raw: str
    text: str
    year_from: Optional[int]
    year_to: Optional[int]
    genres: List[str]
    people: List[str]  # actors/directors names


DECADE_RE = re.compile(r"\b(\d{2})s\b|\b(\d{4})s\b|\b(\d{2})'s\b|\b(\d{4})'s\b", re.I)
EARLY_LATE_RE = re.compile(r"(early|late)\s*(\d{2})s\b|(early|late)\s*(\d{4})s\b", re.I)
YEAR_RANGE_RE = re.compile(r"(\d{4})\s*[-to]+\s*(\d{4})", re.I)
YEAR_SINGLE_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
PERSON_HINT_RE = re.compile(r"(starring|with|featuring|by|directed by)\s+([\w\-\s\.]+)", re.I)


def _normalize_text(q: str) -> str:
    q = q.strip().lower()
    for k, v in GENRE_SYNONYMS.items():
        q = q.replace(k, v)
    return q


def _parse_decade(text: str) -> Optional[Tuple[int, int]]:
    # 90s -> 1990-1999, 80's -> 1980-1989, 2000s -> 2000-2009
    # Check early/late first
    m2 = EARLY_LATE_RE.search(text)
    if m2:
        groups = [g for g in m2.groups() if g]
        if len(groups) >= 2:
            early_late = groups[0]
            v = groups[1]
            v_int = int(v)
            if v_int < 100:  # 2-digit
                start = 1900 + v_int
            else:
                start = v_int
            if early_late.lower() == "early":
                return start, start + 4
            else:
                return start + 5, start + 9
    
    # Check regular decade
    m = DECADE_RE.search(text)
    if m:
        val = next((g for g in m.groups() if g), None)
        if val:
            val_int = int(val)
            if val_int < 100:  # '80s -> 1980-1989
                start = 1900 + val_int
            else:  # '2000s -> 2000-2009
                start = val_int
            return start, start + 9
    return None


def _parse_years(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = YEAR_RANGE_RE.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return a, b
    m2 = _parse_decade(text)
    if m2:
        return m2
    # Single year
    m3 = YEAR_SINGLE_RE.search(text)
    if m3:
        y = int(m3.group(0))
        return y, y
    return None, None


def _extract_people(text: str) -> List[str]:
    people: List[str] = []
    for m in PERSON_HINT_RE.finditer(text):
        name = m.group(2).strip()
        if name:
            people.append(name)
    return people


def _extract_genres(text: str) -> List[str]:
    found: List[str] = []
    for g in ALL_GENRES:
        if g in text:
            found.append(g)
    return sorted(list(set(found)))


def parse_query(q: str) -> ParsedQuery:
    norm = _normalize_text(q)
    yfrom, yto = _parse_years(norm)
    genres = _extract_genres(norm)
    people = _extract_people(q)  # keep capitalization for names
    # Remove extracted hints from text minimally
    cleaned = norm
    for g in genres:
        cleaned = cleaned.replace(g, " ")
    cleaned = PERSON_HINT_RE.sub(" ", cleaned)
    return ParsedQuery(raw=q, text=cleaned.strip(), year_from=yfrom, year_to=yto, genres=genres, people=people)
