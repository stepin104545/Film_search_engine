# Film Search Engine

A natural language movie search system built on the MSRD (Movie Search Ranking Dataset). Search for movies using queries like "comedy films in the 80s starring Eddie Murphy" and get accurate, relevant results with normalized relevance scores.

## Features

- **Natural Language Understanding**: Parse decades (80s, 90s), year ranges (2005-2010), genres, and people
- **Accurate Filtering**: Strict year, genre, and actor/director filters
- **Multi-Signal Scoring**: Combines TF-IDF text matching (40%), fuzzy title matching (20%), and metadata boosts (40%)
- **Normalized Scores**: All relevance scores in 0.0-1.0 range for interpretability
- **Dual Interface**: Command-line tool and Streamlit web application
- **Fast Performance**: Sub-second search over 9,700 movies

## Screenshots

### Streamlit Web Interface

![Search Interface](Screenshot%202025-10-11%20at%201.25.15%20PM.png)
*Search interface with natural language query input*

![Search Results](Screenshot%202025-10-11%20at%201.26.11%20PM.png)
*Search results showing top 5 movies with normalized relevance scores*

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- MSRD dataset (movies.csv.gz)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd repo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download MSRD dataset**:
   - Download from: https://github.com/microsoft/msmarco/tree/main/MSRD
   - Place `movies.csv.gz` in `msrd-dataset/msrd/dataset/` directory
   - Or create directory structure:
```bash
mkdir -p msrd-dataset/msrd/dataset
# Place movies.csv.gz in msrd-dataset/msrd/dataset/
```

4. **Build the search index** (one-time, takes ~30 seconds):
```bash
python -m film_search.cli build-index
```

### Usage

#### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

Features:
- Interactive search box
- Rich result cards with movie metadata
- Top 5 results displayed
- Example queries provided

#### Option 2: Command Line Interface

```bash
python -m film_search.cli search "your query here" -k 5
```

Examples:
```bash
# Search for 80s comedies with Eddie Murphy
python -m film_search.cli search "comedy films in the 80s starring Eddie Murphy" -k 5

# Search for 90s sci-fi movies
python -m film_search.cli search "sci-fi movies from the 90s" -k 5

# Search for action thrillers with Bruce Willis
python -m film_search.cli search "action thriller with Bruce Willis" -k 5
```

## Example Queries

| Query | Description |
|-------|-------------|
| `comedy films in the 80s starring Eddie Murphy` | Decade + genre + actor |
| `sci-fi movies from the 90s` | Decade + genre |
| `action thriller with Bruce Willis` | Multiple genres + actor |
| `war movies about love from early 2000s` | Genre + time period |
| `animated family movies 2005-2010` | Genre + year range |
| `romantic comedies from late 90s` | Genre + time period |

## Architecture

### System Components

```
User Interface
├── CLI (Terminal)
└── Streamlit Web App

Search Engine Core
├── Query Parser (NLP)
├── Filter Engine
├── Scoring Engine
│   ├── Text Score (TF-IDF)
│   ├── Fuzzy Match
│   └── Metadata Boost
└── Ranking Engine

Data Layer
├── Data Loader
├── Indexer (TF-IDF)
└── Index Storage
```

### Query Processing Pipeline

1. **Parse Query**: Extract years, genres, people from natural language
2. **Filter Candidates**: Apply strict filters (year, genre, people)
3. **Score**: Compute relevance using multi-signal approach
4. **Normalize**: Scale scores to 0-1 range
5. **Rank**: Sort by score and return top K results

### Scoring Algorithm

```
Final Score = (Text Score × 0.4) + (Fuzzy Score × 0.2) + (Metadata Score × 0.4)
Normalized Score = Final Score / Max(All Scores)
```

**Components:**
- **Text Score (40%)**: TF-IDF cosine similarity on movie text
- **Fuzzy Score (20%)**: RapidFuzz partial ratio on titles
- **Metadata Score (40%)**: Genre matches (+0.3), people matches (+0.4), year relevance (+0.2), quality boost (+0.15)

## Project Structure

```
film_search_engine_deploy/
├── film_search/              # Core search engine
│   ├── __init__.py
│   ├── data.py              # Data loading and normalization
│   ├── indexer.py           # TF-IDF index building
│   ├── query.py             # Natural language query parsing
│   ├── search.py            # Search and scoring logic
│   └── cli.py               # Command-line interface
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
└── README.md               # This file

Generated files (after build-index):
├── index/
│   └── index.joblib        # Serialized search index
```

## Technical Details

### Query Parser

Extracts structured information from natural language:

**Decades:**
- `80s` → 1980-1989
- `90s` → 1990-1999
- `2000s` → 2000-2009
- `early 2000s` → 2000-2004
- `late 90s` → 1995-1999

**Genres:**
- Direct matching: `comedy`, `action`, `thriller`, `drama`, etc.
- Synonyms: `sci-fi` → `science fiction`, `romcom` → `romance,comedy`

**People:**
- Patterns: `starring X`, `with X`, `featuring X`, `directed by X`
- Example: "starring Tom Hanks" → ["Tom Hanks"]

### Index Building

1. Load MSRD dataset (movies.csv.gz)
2. Normalize data (handle missing values, type conversion)
3. Build weighted text (title×3 + genres×2 + director×2 + actors + overview)
4. Create TF-IDF vectors (1-2 grams, 200k features)
5. Normalize vectors (L2 norm)
6. Save to disk (index/index.joblib)

### Search Process

1. **Parse query** → Extract years, genres, people
2. **Load index** → Deserialize from disk (cached)
3. **Filter** → Apply year/genre/people masks (AND logic)
4. **Score** → Compute text, fuzzy, metadata scores
5. **Combine** → Weighted sum with normalization
6. **Rank** → Sort descending, return top K

## Performance

| Metric | Value |
|--------|-------|
| Dataset Size | 9,700 movies |
| Index Build Time | ~25 seconds |
| Index Size (disk) | ~450 MB |
| Query Latency | <100ms (after index load) |
| Memory Usage | ~550 MB |
| Score Range | 0.0 - 1.0 (normalized) |

## Dependencies

- **pandas** (≥2.0.0): Data manipulation
- **scikit-learn** (≥1.3.0): TF-IDF vectorization
- **numpy** (≥1.24.0, <2.0.0): Numerical operations
- **rapidfuzz** (≥3.0.0): Fuzzy string matching
- **joblib** (≥1.3.0): Index serialization
- **rich** (≥13.0.0): CLI formatting
- **streamlit** (≥1.28.0): Web application

## Design Decisions

### Why TF-IDF over Embeddings?

- **Speed**: No API calls, runs locally
- **Simplicity**: Well-understood, easy to debug
- **No Dependencies**: No API keys or large model downloads
- **Effective**: Good for keyword-based queries
- **Interpretable**: Can see which terms matched

**Trade-off**: Sacrificed semantic understanding for speed and simplicity. For production, would implement hybrid approach (TF-IDF + embeddings).

### Why Multi-Signal Scoring?

Single signal (TF-IDF alone) misses important information:
- Text score: Captures topical relevance
- Fuzzy score: Handles typos and partial matches
- Metadata score: Leverages structured data (genres, actors, ratings)

Combined approach is more robust and accurate.

### Why Normalize Scores?

Raw scores (0.1-1.3) are not intuitive. Normalized scores (0-1) are:
- **Interpretable**: 1.0 = best match, 0.5 = moderate match
- **Comparable**: Can compare across queries
- **User-friendly**: "90% relevance" makes sense

## Limitations

- **No Semantic Understanding**: Keyword-based, doesn't understand "movies about redemption"
- **Limited Synonyms**: Small synonym dictionary (4 entries)
- **No Learning-to-Rank**: Fixed scoring weights, not learned from data
- **Single Machine**: Not designed for distributed deployment
- **No Personalization**: Same results for all users

## Future Improvements

### Short Term
1. Expand synonym dictionary (100+ entries)
2. Add query spell-check and correction
3. Implement caching for popular queries
4. Add more genre mappings

### Medium Term
1. Add semantic embeddings (sentence-transformers)
2. Implement hybrid retrieval (TF-IDF + embeddings)
3. Add Named Entity Recognition for better people extraction
4. Precompute filter indices for faster filtering

### Long Term
1. Learning-to-rank with MSRD relevance labels
2. Evaluate with nDCG, MAP metrics
3. Distributed architecture (Elasticsearch + FAISS)
4. User personalization and feedback loop

## Evaluation

### Current
Manual testing with example queries, verified:
- Filters work correctly (80s → only 1980-1989)
- Scores normalized (0-1 range)
- Results relevant to query

### Proper Evaluation (Future)
1. Use MSRD queries.csv.gz (28k queries with relevance labels)
2. Compute metrics:
   - **nDCG@5**: Normalized Discounted Cumulative Gain
   - **MAP**: Mean Average Precision
   - **MRR**: Mean Reciprocal Rank
3. Compare against BM25 baseline
4. A/B testing with real users

## Troubleshooting

### Index not found error
```
FileNotFoundError: Index not found at index/index.joblib
```
**Solution**: Run `python -m film_search.cli build-index` first

### CSV parsing error
```
ParserError: Error tokenizing data
```
**Solution**: Ensure movies.csv.gz is in correct location: `msrd-dataset/msrd/dataset/movies.csv.gz`

### Import errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Slow first query
First query loads index from disk (~50ms). Subsequent queries are cached and faster (<10ms).

## License

This project is for educational and demonstration purposes.

## Dataset

MSRD (Movie Search Ranking Dataset) by Microsoft Research:
- Source: https://github.com/microsoft/msmarco/tree/main/MSRD
- Contains: ~9,700 movies with metadata (title, overview, genres, cast, ratings)
- Format: TSV (tab-separated values) compressed with gzip

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with Python, scikit-learn, and Streamlit**
