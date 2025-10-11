import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .indexer import build_index
from .query import parse_query
from .search import search


console = Console()


def cmd_build(args: argparse.Namespace) -> None:
    console.print("[bold green]Building index...[/bold green]")
    build_index()
    console.print("[bold green]Index built and saved to ./index/index.joblib[/bold green]")


def _print_results(results):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right")
    table.add_column("Title")
    table.add_column("Year", justify="right")
    table.add_column("Genres")
    table.add_column("Director")
    table.add_column("Actors")
    table.add_column("Score", justify="right")
    for i, r in enumerate(results, 1):
        table.add_row(str(i), r.title, str(r.year or ""), r.genres, r.director, r.actors[:60] + ("..." if len(r.actors) > 60 else ""), f"{r.score:.3f}")
    console.print(table)


def cmd_search(args: argparse.Namespace) -> None:
    q = parse_query(args.query)
    results = search(q, top_k=args.k)
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    _print_results(results)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("film-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Build the TF-IDF index from MSRD movies")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="Search movies with a natural language query")
    s.add_argument("query", type=str, help="e.g., 'sci-fi movies from the 90s with Tom Hanks'")
    s.add_argument("-k", type=int, default=20, help="number of results")
    s.set_defaults(func=cmd_search)

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
