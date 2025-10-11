import streamlit as st
from pathlib import Path

from film_search.indexer import load_index
from film_search.query import parse_query
from film_search.search import search


# Page config
st.set_page_config(
    page_title="Film Search Engine",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .movie-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #ff4b4b;
    }
    .movie-title {
        font-size: 24px;
        font-weight: bold;
        color: #262730;
        margin-bottom: 10px;
    }
    .movie-meta {
        font-size: 14px;
        color: #555;
        margin: 5px 0;
    }
    .score-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    .search-box {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)


# Title and description
st.title("Film Search Engine")
st.markdown("Search for movies using natural language queries like *'sci-fi movies from the 90s with Tom Hanks'*")
st.markdown("**Top 5 most relevant results** will be shown with enhanced scoring")

# Check if index exists
index_path = Path("index/index.joblib")
if not index_path.exists():
    st.error("⚠️ Index not found! Please build the index first by running: `python -m film_search.cli build-index`")
    st.stop()

# Search input
query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., comedy films in the 80s starring Eddie Murphy",
    key="search_query",
    label_visibility="collapsed"
)

# Search button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("Search", use_container_width=True, type="primary")

# Perform search
if search_button and query:
    with st.spinner("Searching..."):
        try:
            parsed = parse_query(query)
            results = search(parsed, top_k=5)
            
            if not results:
                st.warning("No results found. Try a different query!")
            else:
                st.success(f"Found {len(results)} top results")
                
                # Display results
                for i, result in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">#{i} {result.title}</div>
                            <div class="movie-meta">
                                <strong>Year:</strong> {result.year if result.year else 'N/A'}
                            </div>
                            <div class="movie-meta">
                                <strong>Genres:</strong> {result.genres if result.genres else 'N/A'}
                            </div>
                            <div class="movie-meta">
                                <strong>Director:</strong> {result.director if result.director else 'N/A'}
                            </div>
                            <div class="movie-meta">
                                <strong>Cast:</strong> {result.actors[:100] + ('...' if len(result.actors) > 100 else '') if result.actors else 'N/A'}
                            </div>
                            <span class="score-badge">Relevance Score: {result.score:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Example queries
with st.expander("Example Queries"):
    st.markdown("""
    - `sci-fi movies from the 90s with Tom Hanks`
    - `comedy films in the 80s starring Eddie Murphy`
    - `war movies about love from early 2000s`
    - `animated family movies 2005-2010`
    - `action thriller with Bruce Willis`
    - `romantic comedies from late 90s`
    """)

# Footer
st.markdown("---")
st.markdown("Built using Streamlit | Dataset: MSRD (Movie Search Ranking Dataset)")
