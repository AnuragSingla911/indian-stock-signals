"""CLI to refresh the on-disk news archive from Finnhub."""

from __future__ import annotations

import argparse
import logging

from .config import CONFIG
from .news_archive import finnhub_api_key, get_news_archive
from .universe import load_universe

log = logging.getLogger("iss.news_archive_cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh historical news archive (Finnhub) for backtestable sentiment"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=CONFIG.news_archive_days,
        help="how many calendar days of history to fetch",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="optional subset of symbols (default: full universe)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not finnhub_api_key():
        log.info("FINNHUB_API_KEY not set; using Google News RSS for NSE symbols")

    symbols = args.symbols or [s.symbol for s in load_universe()]
    archive = get_news_archive()
    log.info("Refreshing news archive for %d symbols (%d days)", len(symbols), args.days)
    added = archive.refresh_all(symbols, days_back=args.days)
    total_new = sum(added.values())
    populated = sum(1 for v in added.values() if v > 0)
    print(f"Archive refresh complete: {total_new} new articles across {populated} symbols")


if __name__ == "__main__":
    main()
