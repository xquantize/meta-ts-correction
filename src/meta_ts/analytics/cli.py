from __future__ import annotations

import argparse
from pathlib import Path

from meta_ts.analytics.factory import get_analytics
from meta_ts.results.paths import tables_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-tables",
        description="Query run artifacts and export paper-ready tables",
    )
    parser.add_argument("--base", default="outputs", help="Output root directory")
    parser.add_argument(
        "--backend",
        default="duckdb",
        choices=["duckdb", "lancedb"],
        help="Analytics backend (duckdb now; lancedb reserved for embeddings)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="CSV path for leaderboard (default: outputs/tables/leaderboard.csv)",
    )
    parser.add_argument("--list-runs", action="store_true", help="Print known runs and exit")
    args = parser.parse_args(argv)

    store = get_analytics(backend=args.backend, base=args.base)

    if args.list_runs:
        runs = store.list_runs()
        if runs.empty:
            print("no runs found")
            return 0
        cols = [c for c in ["run_id", "name", "model", "status", "created_at"] if c in runs.columns]
        print(runs[cols].to_string(index=False))
        return 0

    board = store.leaderboard()
    if board.empty:
        print("no scores found")
        return 0

    print(board.to_string(index=False))
    out = Path(args.export) if args.export else tables_root(args.base) / "leaderboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
