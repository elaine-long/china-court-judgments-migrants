"""Readers for the working data layer."""
import duckdb
import pandas as pd

from src import paths


def load_meta(columns=None) -> pd.DataFrame:
    return pd.read_parquet(paths.META, columns=columns)


def load_texts(doc_ids, columns=None, path=None) -> pd.DataFrame:
    """Load rows of docs_text.parquet for a list of doc_ids (returned sorted by doc_id).

    Uses duckdb so that parquet row-group statistics on doc_id prune the scan.
    """
    path = str(path or paths.DOCS_TEXT)
    ids = sorted({int(i) for i in doc_ids})
    if not ids:
        return pd.DataFrame()
    cols = "*" if columns is None else ", ".join(
        dict.fromkeys(["doc_id", *columns]))
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    try:
        con.register("ids", pd.DataFrame({"doc_id": ids}))
        q = (f"SELECT {cols} FROM read_parquet('{path}') "
             f"WHERE doc_id IN (SELECT doc_id FROM ids) ORDER BY doc_id")
        return con.execute(q).df()
    finally:
        con.close()
