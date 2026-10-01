"""Central registry of file paths. All other code should import paths from here.

Raw data live outside the repo and are READ-ONLY.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # paperB/
RAW_ROOT = ROOT.parent / "中国裁判文书犯罪数据-zhang, yan. (2024)"

# Raw data (read-only)
RAW_MERGED_FULL = RAW_ROOT / "Section 5 Long" / "Data" / "ChinaCrime_Merged_Full.csv"
RAW_CASE_INFO = RAW_ROOT / "Section 5 Long" / "Data" / "case_information.csv"
RAW_ZHANG = RAW_ROOT / "ChinaCrimeDatas.csv"
RAW_EXAMPLE_DIR = RAW_ROOT / "Data Example"
RAW_EXAMPLE_MERGED = RAW_EXAMPLE_DIR / "ChinaCrime_Merged_Full Example.csv"
RAW_EXAMPLE_CASE_INFO = RAW_EXAMPLE_DIR / "case_information example.csv"
RAW_OLD_EXTRACTION_NB = RAW_ROOT / "0513 Final Paper" / "Code" / "Section 2.2.1 Code.ipynb"

# Derived data
DATA = ROOT / "data"
DOCS_TEXT = DATA / "docs_text.parquet"
OLD_FEATURES = DATA / "old_features.parquet"
META = DATA / "meta.parquet"
THEFT_CANDIDATES = DATA / "theft_candidates.parquet"

# Outputs
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"

SEED = 20260929
