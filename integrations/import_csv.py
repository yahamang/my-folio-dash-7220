#!/usr/bin/env python3
"""
Mirae Asset MTS/HTS CSV Import
Reads portfolio export from Mirae Asset and updates config.json holdings.

How to export from Mirae Asset MTS:
  1. Open MTS → 투자현황 (Portfolio)
  2. Tap ⋮ menu → 내보내기 / 엑셀저장
  3. Save the file and rename it to:
       ~/investment/integrations/mirae_domestic.csv   (ISA / pension — Korean stocks)
       ~/investment/integrations/mirae_overseas.csv   (overseas stocks)

CSV column mapping (Mirae Asset standard export):
  Korean:  종목코드, 종목명, 보유수량, 매입금액, 현재가, 평가금액, 수익률(%)
  Overseas: 종목코드, 종목명, 보유수량, 매입금액(원), 현재가(USD), 평가금액(원), 수익률(%)

Usage:
  python3 ~/investment/integrations/import_csv.py [--dry-run]

  --dry-run : print parsed data without modifying config.json
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR    = Path.home() / "investment"
CONFIG_PATH = BASE_DIR / "config.json"
CSV_DIR     = BASE_DIR / "integrations"

DOMESTIC_CSV = CSV_DIR / "mirae_domestic.csv"
OVERSEAS_CSV = CSV_DIR / "mirae_overseas.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsers
# ─────────────────────────────────────────────────────────────────────────────
def _find_col(headers: list[str], candidates: list[str]) -> int | None:
    """Find column index by trying multiple candidate names."""
    for c in candidates:
        for i, h in enumerate(headers):
            if c in h:
                return i
    return None


def parse_domestic_csv(path: Path) -> list[dict]:
    """
    Parse Mirae Asset domestic (Korean) holdings CSV.
    Returns list of {ticker, name, shares, avg_cost_krw}
    """
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)

        idx_code   = _find_col(headers, ["종목코드", "코드"])
        idx_name   = _find_col(headers, ["종목명", "종목"])
        idx_shares = _find_col(headers, ["보유수량", "수량"])
        idx_cost   = _find_col(headers, ["매입금액", "매입"])

        if None in (idx_code, idx_name, idx_shares, idx_cost):
            raise ValueError(
                f"Required columns not found in {path.name}.\n"
                f"Headers found: {headers}\n"
                "Expected: 종목코드, 종목명, 보유수량, 매입금액"
            )

        for row in reader:
            if not row or not row[idx_code].strip():
                continue
            code = row[idx_code].strip()
            # Add .KS suffix if not present
            ticker = code if ("." in code) else f"{code}.KS"
            shares = float(row[idx_shares].replace(",", "").strip() or 0)
            cost   = float(row[idx_cost].replace(",", "").strip() or 0)
            if shares > 0:
                rows.append({
                    "ticker":       ticker,
                    "name":         row[idx_name].strip(),
                    "shares":       shares,
                    "avg_cost_krw": cost,
                })
    return rows


def parse_overseas_csv(path: Path) -> list[dict]:
    """
    Parse Mirae Asset overseas holdings CSV.
    Returns list of {ticker, name, shares, avg_cost_krw}
    """
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)

        idx_code   = _find_col(headers, ["종목코드", "티커", "Ticker"])
        idx_name   = _find_col(headers, ["종목명", "종목"])
        idx_shares = _find_col(headers, ["보유수량", "수량"])
        idx_cost   = _find_col(headers, ["매입금액", "매입"])

        if None in (idx_code, idx_name, idx_shares, idx_cost):
            raise ValueError(
                f"Required columns not found in {path.name}.\n"
                f"Headers found: {headers}\n"
                "Expected: 종목코드/티커, 종목명, 보유수량, 매입금액"
            )

        for row in reader:
            if not row or not row[idx_code].strip():
                continue
            ticker = row[idx_code].strip()
            shares = float(row[idx_shares].replace(",", "").strip() or 0)
            cost   = float(row[idx_cost].replace(",", "").strip() or 0)
            if shares > 0:
                rows.append({
                    "ticker":       ticker,
                    "name":         row[idx_name].strip(),
                    "shares":       shares,
                    "avg_cost_krw": cost,
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Config updater
# ─────────────────────────────────────────────────────────────────────────────
def update_config(imported: dict[str, list], dry_run: bool = False) -> None:
    """
    Update config.json holdings with imported CSV data.

    imported: {"pension": [...], "isa": [...], "overseas": [...]}
    Matching is done by ticker. Shares and avg_cost_krw are updated.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    changes = []
    for acc_id, rows in imported.items():
        if acc_id not in config["portfolio"]["accounts"]:
            print(f"  ⚠ Account '{acc_id}' not found in config.json — skipped")
            continue

        holdings = config["portfolio"]["accounts"][acc_id]["holdings"]
        by_ticker = {h["ticker"]: h for h in holdings}

        for row in rows:
            t = row["ticker"]
            if t in by_ticker:
                h = by_ticker[t]
                old_shares = h.get("shares", 0)
                old_cost   = h.get("avg_cost_krw", 0)
                h["shares"]       = row["shares"]
                h["avg_cost_krw"] = row["avg_cost_krw"]
                changes.append(
                    f"  {acc_id}/{t}: shares {old_shares}→{row['shares']}, "
                    f"avg_cost_krw {old_cost:,.0f}→{row['avg_cost_krw']:,.0f}"
                )
            else:
                print(f"  ℹ {acc_id}/{t} not in config.json — add manually if needed")

    if not changes:
        print("No changes detected.")
        return

    print("Changes to apply:")
    for c in changes:
        print(c)

    if dry_run:
        print("\n[dry-run] config.json not modified.")
        return

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n✅ config.json updated ({len(changes)} holdings)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv
    imported: dict[str, list] = {}
    found_any = False

    # Domestic CSV → try to map to pension + isa accounts
    if DOMESTIC_CSV.exists():
        found_any = True
        print(f"📄 Reading {DOMESTIC_CSV.name}...")
        rows = parse_domestic_csv(DOMESTIC_CSV)
        print(f"   {len(rows)} holdings parsed")

        # Split into pension vs isa by matching existing tickers
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

        pension_tickers = {h["ticker"] for h in config["portfolio"]["accounts"]["pension"]["holdings"]}
        isa_tickers     = {h["ticker"] for h in config["portfolio"]["accounts"]["isa"]["holdings"]}

        pension_rows, isa_rows, unmatched = [], [], []
        for row in rows:
            if row["ticker"] in pension_tickers:
                pension_rows.append(row)
            elif row["ticker"] in isa_tickers:
                isa_rows.append(row)
            else:
                unmatched.append(row)

        if pension_rows:
            imported["pension"] = pension_rows
        if isa_rows:
            imported["isa"] = isa_rows
        if unmatched:
            print(f"  ⚠ Unmatched tickers: {[r['ticker'] for r in unmatched]}")

    # Overseas CSV → overseas account
    if OVERSEAS_CSV.exists():
        found_any = True
        print(f"📄 Reading {OVERSEAS_CSV.name}...")
        rows = parse_overseas_csv(OVERSEAS_CSV)
        print(f"   {len(rows)} holdings parsed")
        imported["overseas"] = rows

    if not found_any:
        print(
            "No CSV files found.\n\n"
            "Export from Mirae Asset MTS:\n"
            "  1. 투자현황 → ⋮ → 엑셀저장\n"
            f"  2. Save as: {DOMESTIC_CSV}\n"
            f"         or: {OVERSEAS_CSV}"
        )
        return

    if imported:
        update_config(imported, dry_run=dry_run)


if __name__ == "__main__":
    main()
