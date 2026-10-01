"""Parsing of Chinese criminal case numbers (案号).

New format (2016+):  (2018)粤0902刑初370号 -> abbr 粤, court code 0902, kind 刑初
Old format (<=2015): (2015)兴刑初字第153号 -> no province abbreviation, kind 刑初
"""
import re

import pandas as pd

PROV_ABBR = {
    "京": "北京市", "津": "天津市", "沪": "上海市", "渝": "重庆市",
    "冀": "河北省", "晋": "山西省", "辽": "辽宁省", "吉": "吉林省", "黑": "黑龙江省",
    "苏": "江苏省", "浙": "浙江省", "皖": "安徽省", "闽": "福建省", "赣": "江西省",
    "鲁": "山东省", "豫": "河南省", "鄂": "湖北省", "湘": "湖南省", "粤": "广东省",
    "琼": "海南省", "川": "四川省", "黔": "贵州省", "云": "云南省", "陕": "陕西省",
    "甘": "甘肃省", "青": "青海省", "台": "台湾省",
    "内": "内蒙古自治区", "桂": "广西壮族自治区", "藏": "西藏自治区",
    "宁": "宁夏回族自治区", "新": "新疆维吾尔自治区",
    "港": "香港特别行政区", "澳": "澳门特别行政区",
}

# "(2018)" or "（2018）" then one province abbreviation then a numeric court code
_NEW_FMT = re.compile(r"^\s*[(（]\s*\d{4}\s*[)）]\s*(\D)(\d+)")
# kind segment: from the first 刑 up to the next digit (works with or without the year prefix)
_KIND_SEG = re.compile(r"(刑[^\d号]*)")


def parse_prov_code(case_number):
    """Return (abbr, court_code) for new-format case numbers, else (None, None)."""
    if not isinstance(case_number, str):
        return None, None
    m = _NEW_FMT.match(case_number)
    if not m or m.group(1) not in PROV_ABBR:
        return None, None
    return m.group(1), m.group(2)


def parse_doc_kind(case_number):
    """Classify into 刑初 / 刑终 / 刑更 / 其他."""
    if not isinstance(case_number, str):
        return "其他"
    m = _KIND_SEG.search(case_number)
    if not m:
        return "其他"
    seg = m.group(1)
    if "终" in seg:
        return "刑终"
    if "更" in seg:
        return "刑更"
    if "初" in seg:
        return "刑初"
    return "其他"


def parse_case_numbers(s: pd.Series) -> pd.DataFrame:
    """Vectorised-ish parsing of a Series of case numbers."""
    uniq = pd.Series(s.dropna().unique())
    abbr, code, kind = {}, {}, {}
    for cn in uniq:
        a, c = parse_prov_code(cn)
        abbr[cn], code[cn], kind[cn] = a, c, parse_doc_kind(cn)
    out = pd.DataFrame({
        "court_prov_abbr": s.map(abbr),
        "court_code": s.map(code),
        "doc_kind": s.map(kind).fillna("其他"),
    }, index=s.index)
    out["court_province"] = out["court_prov_abbr"].map(PROV_ABBR)
    return out
