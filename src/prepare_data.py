"""Prepare the canonical annotated dataset used by the analysis notebooks.

Reads the raw annotation workbook, restores text encoding, checks the label
columns against the annotation scheme, and writes a single canonical CSV.

Usage:
    python src/prepare_data.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from ast import literal_eval
from csv import DictReader, field_size_limit
from pathlib import Path

import ftfy
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "Annotated_dataset.xlsx"
CORPUS_CSV = ROOT / "data" / "dataset.english.csv"
OUT_CSV = ROOT / "data" / "annotated_clean.csv"

TEXT_COLS = ["target_tweet", "authentic_reply"]
LABEL_COLS = ["STANCE", "ACTION", "PERSONALNESS", "POLITENESS"]

# The annotation scheme: the permitted values for each dimension.
LABEL_DOMAINS = {
    "STANCE": {"SUPPORT", "OPPOSE", "NEUTRAL"},
    "ACTION": {"STATEMENT", "QUESTION", "COMMAND", "REACTION"},
    "PERSONALNESS": {"GENERAL", "PERSONAL"},
    "POLITENESS": {"NORMAL", "POLITE", "RUDE"},
}


def match_key(text: str) -> str:
    """Aggressively normalised form of a string, used only for joining."""
    text = ftfy.fix_text(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"\s+", " ", text).strip().lower()


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the word sets of two normalised strings."""
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def load_corpus(path: Path) -> tuple[dict, dict]:
    """Index the source corpus by (tweet, reply) pair and by reply alone."""
    field_size_limit(10**9)

    def last_user_turn(prompt: str) -> str | None:
        try:
            for turn in reversed(literal_eval(prompt)):
                if turn.get("role") == "user":
                    return turn.get("content")
        except (ValueError, SyntaxError):
            pass
        return None

    by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    by_reply: dict[str, list[tuple[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in DictReader(handle):
            tweet = last_user_turn(row["prompt"])
            if not tweet:
                continue
            reply = row["authentic_reply"]
            k_tweet, k_reply = match_key(tweet), match_key(reply)
            by_pair[(k_tweet, k_reply)] = (tweet, reply)
            by_reply.setdefault(k_reply, []).append((tweet, reply))
    return by_pair, by_reply


def restore_text(df: pd.DataFrame, by_pair: dict, by_reply: dict) -> tuple[pd.DataFrame, dict]:
    """Recover the original wording of each pair from the source corpus.

    A pair is restored when both its tweet and its reply identify the same
    corpus record. Where only the reply matches, the record is used solely if
    it is unambiguous and its tweet is substantially the same text. Anything
    else falls back to codec repair.
    """
    df = df.copy()
    stats = {"pair": 0, "reply": 0, "repaired": 0}

    for i, row in df.iterrows():
        k_tweet, k_reply = match_key(row["target_tweet"]), match_key(row["authentic_reply"])
        source = by_pair.get((k_tweet, k_reply))

        if source is not None:
            stats["pair"] += 1
        else:
            candidates = by_reply.get(k_reply, [])
            if len(candidates) == 1 and token_overlap(k_tweet, match_key(candidates[0][0])) >= 0.6:
                source = candidates[0]
                stats["reply"] += 1

        if source is not None:
            df.at[i, "target_tweet"], df.at[i, "authentic_reply"] = source
        else:
            stats["repaired"] += 1
            for col in TEXT_COLS:
                df.at[i, col] = ftfy.fix_text(str(row[col]))

    for col in TEXT_COLS:
        df[col] = df[col].str.strip()
    return df, stats


def normalise_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Place each annotation in the dimension its value belongs to.

    PERSONALNESS and POLITENESS share a row-level pair of values; where the two
    are transposed relative to the scheme, they are restored to their own
    columns. Returns the frame and the number of rows adjusted.
    """
    df = df.copy()
    for col in LABEL_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper()

    transposed = df["PERSONALNESS"].isin(LABEL_DOMAINS["POLITENESS"]) & df[
        "POLITENESS"
    ].isin(LABEL_DOMAINS["PERSONALNESS"])

    df.loc[transposed, ["PERSONALNESS", "POLITENESS"]] = df.loc[
        transposed, ["POLITENESS", "PERSONALNESS"]
    ].values

    return df, int(transposed.sum())


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the prepared frame violates the annotation scheme."""
    problems = []

    if df.empty:
        problems.append("dataset is empty")

    for col in TEXT_COLS + LABEL_COLS:
        missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if missing:
            problems.append(f"{col}: {missing} missing values")

    for col, domain in LABEL_DOMAINS.items():
        unexpected = sorted(set(df[col]) - domain)
        if unexpected:
            problems.append(f"{col}: values outside the scheme: {unexpected}")

    duplicates = df.duplicated(subset=TEXT_COLS).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate (target_tweet, authentic_reply) pairs")

    if problems:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(problems))


def main() -> int:
    if not RAW_XLSX.exists():
        print(f"error: {RAW_XLSX} not found", file=sys.stderr)
        return 1

    df = pd.read_excel(RAW_XLSX)
    expected = set(TEXT_COLS + LABEL_COLS)
    if not expected.issubset(df.columns):
        print(f"error: workbook is missing columns {sorted(expected - set(df.columns))}", file=sys.stderr)
        return 1
    df = df[TEXT_COLS + LABEL_COLS]
    n_input = len(df)

    by_pair, by_reply = load_corpus(CORPUS_CSV) if CORPUS_CSV.exists() else ({}, {})
    df, stats = restore_text(df, by_pair, by_reply)
    df, adjusted = normalise_labels(df)

    validate(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    matched = stats["pair"] + stats["reply"]
    print(f"rows in               : {n_input}")
    print(f"rows written          : {len(df)}")
    print(f"text matched to corpus: {matched}/{n_input} "
          f"(pair {stats['pair']}, reply {stats['reply']})")
    print(f"text repaired in place: {stats['repaired']}")
    print(f"label columns aligned : {adjusted} rows")
    print(f"output                : {OUT_CSV.relative_to(ROOT)}\n")

    for col in LABEL_COLS:
        counts = df[col].value_counts()
        summary = "  ".join(f"{k} {v} ({v / len(df):.1%})" for k, v in counts.items())
        print(f"{col:13}: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
