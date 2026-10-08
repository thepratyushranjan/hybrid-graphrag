"""Read rows straight out of a MySQL dump (`INSERT INTO `t` (cols) VALUES (...), (...);`) without a MySQL server.

Handles quoted strings with '' and backslash escapes (multi-line values included), NULL and numbers.
Only the tables asked for are materialised, so a 76 MB dump streams through in a few seconds.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_INSERT = re.compile(r"INSERT INTO `(\w+)` \(([^)]*)\) VALUES\s*", re.S)
_ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
_NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?$")


class DumpParseError(ValueError):
    pass


def _scalar(token: str) -> Any:
    if token.upper() == "NULL":
        return None
    if _NUMBER.match(token):
        return float(token) if any(c in token for c in ".eE") else int(token)
    return token


def _values(text: str, pos: int) -> tuple[list[list[Any]], int]:
    """Parse `(v, v, ...), (...);` starting at `pos`; returns the rows and the position after the `;`."""
    rows: list[list[Any]] = []
    n = len(text)
    while True:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n:
            raise DumpParseError("unexpected end of dump inside VALUES")
        if text[pos] == ";":
            return rows, pos + 1
        if text[pos] != "(":
            raise DumpParseError(f"expected '(' at offset {pos}, got {text[pos]!r}")
        pos += 1
        row: list[Any] = []
        while True:
            while text[pos] in " \t\r\n":
                pos += 1
            if text[pos] == "'":
                pos += 1
                parts: list[str] = []
                start = pos
                while True:
                    ch = text[pos]
                    if ch == "\\":
                        parts.append(text[start:pos])
                        nxt = text[pos + 1]
                        parts.append(_ESCAPES.get(nxt, nxt))
                        pos += 2
                        start = pos
                    elif ch == "'":
                        if text[pos + 1] == "'":  # '' = escaped quote
                            parts.append(text[start : pos + 1])
                            pos += 2
                            start = pos
                        else:
                            parts.append(text[start:pos])
                            pos += 1
                            break
                    else:
                        pos += 1
                row.append("".join(parts))
            else:
                end = pos
                while text[end] not in ",)":
                    end += 1
                row.append(_scalar(text[pos:end].strip()))
                pos = end
            while text[pos] in " \t\r\n":
                pos += 1
            if text[pos] == ",":
                pos += 1
                continue
            if text[pos] == ")":
                pos += 1
                break
            raise DumpParseError(f"expected ',' or ')' at offset {pos}")
        rows.append(row)


def iter_rows(text: str, tables: set[str] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """(table, row-as-dict) for every INSERT row in the dump, optionally only for `tables`."""
    pos = 0
    while (m := _INSERT.search(text, pos)) is not None:
        table = m.group(1)
        columns = [c.strip().strip("`") for c in m.group(2).split(",")]
        rows, pos = _values(text, m.end())
        if tables is not None and table not in tables:
            continue
        for values in rows:
            if len(values) != len(columns):
                raise DumpParseError(f"{table}: {len(values)} values for {len(columns)} columns")
            yield table, dict(zip(columns, values, strict=True))


def load_tables(source: str | Path | bytes, tables: set[str]) -> dict[str, list[dict[str, Any]]]:
    """All rows of `tables`, keyed by table name (missing tables map to [])."""
    if isinstance(source, bytes):
        text = source.decode("utf-8", errors="replace")
    else:
        text = Path(source).read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in tables}
    for table, row in iter_rows(text, tables):
        out[table].append(row)
    return out
