"""Gazetteer of county-level-and-above divisions (1981-2025, with history) and place resolution.

Source: yescallop/areacodes (CC0), result.csv, pinned commit in data/external/areacodes_VERSION.txt.
Fetch (data/ is not in git):
    curl -o data/external/areacodes_result.csv \
      https://raw.githubusercontent.com/yescallop/areacodes/e2c2b2e1fb4ce1fb76e879aa0b80e229adaf950c/result.csv
Every record is normalised to its 2016 successor (`code_2016`), so historical names (蓟县, 襄樊市,
巢湖地区的庐江县) map to the prefecture/province they belonged to during the study window.
"""
import re
from functools import lru_cache

import pandas as pd

from src import paths

RAW = paths.DATA / "external" / "areacodes_result.csv"
GAZ = paths.DATA / "gazetteer.parquet"
REF_YEAR = 2016
WINDOW = (2012, 2020)
MUNI_PREFIX = {"11": "北京市", "12": "天津市", "31": "上海市", "50": "重庆市"}
PROV_SHORT = {"内蒙古自治区": ["内蒙古", "内蒙"], "广西壮族自治区": ["广西"], "西藏自治区": ["西藏"],
              "宁夏回族自治区": ["宁夏"], "新疆维吾尔自治区": ["新疆"], "香港特别行政区": ["香港"],
              "澳门特别行政区": ["澳门"], "黑龙江省": ["黑龙江"]}
LOCAL_WORDS = ("本市", "本区", "本县", "本省", "本镇", "本乡", "本地", "本辖区")
_ETHNIC_AUTONOMY = re.compile(r"(?:[一-龥]{1,6}?族)+自治(?:县|州|旗)$|各族自治县$")
_PREFIX_JUNK = re.compile(r"^(?:中华人民共和国|中国)")


# ----------------------------------------------------------------------------- build
def _short_names(name, level):
    out = []
    if level == "省级":
        out += PROV_SHORT.get(name, [])
        if name.endswith(("省", "市")) and len(name) >= 3:
            out.append(name[:-1])
        return out
    base = _ETHNIC_AUTONOMY.sub("", name)
    if base != name and len(base) >= 2:
        out.append(base)
    for suf in ("地区", "林区", "特区", "市", "县", "区", "旗", "盟"):
        if base.endswith(suf) and len(base) - len(suf) >= 2:
            out.append(base[: -len(suf)])
            break
    return [x for x in dict.fromkeys(out) if x != name]


def build_gazetteer():
    g = pd.read_csv(RAW, dtype=str)
    g.columns = [c.strip("﻿") for c in g.columns]
    g = g.rename(columns={"代码": "code", "一级行政区": "province_at", "二级行政区": "parent_at",
                          "名称": "full_name", "级别": "level", "状态": "status", "启用时间": "valid_from",
                          "变更/弃用时间": "valid_to", "新代码": "new_codes"})
    g["valid_from"] = g["valid_from"].astype(int)
    g["valid_to"] = pd.to_numeric(g["valid_to"]).astype("Int64")

    def valid_in(r, y):
        return r.valid_from <= y and (pd.isna(r.valid_to) or y < r.valid_to)

    by_code = {}
    for r in g.itertuples():
        by_code.setdefault(r.code, []).append(r)

    def rec_at(code, y):
        for r in by_code.get(code, []):
            if valid_in(r, y):
                return r
        return None

    @lru_cache(maxsize=None)
    def successor(code, depth=0):
        """2016 code for an old code, following 新代码 links."""
        if rec_at(code, REF_YEAR) is not None:
            return code
        if depth > 10:
            return None
        for r in sorted(by_code.get(code, []), key=lambda x: -x.valid_from):
            if isinstance(r.new_codes, str) and r.new_codes:
                first = re.sub(r"\[\d{4}\]", "", r.new_codes.split(";")[0])
                s = successor(first, depth + 1)
                if s:
                    return s
        return None

    def prefecture_of(code16):
        r = rec_at(code16, REF_YEAR)
        if r is None:
            return None, None
        if r.level == "省级":
            return None, None
        if code16[:2] in MUNI_PREFIX:
            return MUNI_PREFIX[code16[:2]], code16[:2] + "0000"
        if r.level == "地级":
            return r.full_name, code16
        p = rec_at(code16[:4] + "00", REF_YEAR)
        if p is not None and p.level == "地级":
            return p.full_name, p.code
        return r.full_name, code16  # 省直辖县级行政单位 (仙桃市, 济源市 …)

    rows = []
    for r in g.itertuples():
        c16 = successor(r.code) if r.level != "省级" else r.code
        prov_rec = rec_at((c16 or r.code)[:2] + "0000", REF_YEAR)
        province = prov_rec.full_name if prov_rec is not None else r.province_at
        pref, pref_code = prefecture_of(c16) if c16 else (None, None)
        in_window = r.valid_from <= WINDOW[1] and (pd.isna(r.valid_to) or r.valid_to > WINDOW[0])
        vt = "" if pd.isna(r.valid_to) else str(int(r.valid_to) - 1)
        base = {"full_name": r.full_name, "level": {"省级": "province", "地级": "prefecture", "县级": "county"}[r.level],
                "code": r.code, "code_2016": c16,
                "county_code": c16 if r.level == "县级" else None,
                "prefecture": pref, "prefecture_code": pref_code,
                "province": province, "province_code": (c16 or r.code)[:2] + "0000",
                "valid_from": r.valid_from, "valid_to": r.valid_to, "valid_years": f"{r.valid_from}-{vt}",
                "in_window": bool(in_window)}
        rows.append({"name": r.full_name, "is_short": False, **base})
        for s in _short_names(r.full_name, r.level):
            rows.append({"name": s, "is_short": True, **base})
    out = pd.DataFrame(rows)
    out.to_parquet(GAZ, index=False)
    return out


# ----------------------------------------------------------------------------- resolve
_INDEX = None


def _load():
    global _INDEX
    if _INDEX is None:
        gz = pd.read_parquet(GAZ) if GAZ.exists() else build_gazetteer()
        idx = {}
        for r in gz.to_dict("records"):
            idx.setdefault(r["name"], []).append(r)
        _INDEX = (idx, max(len(k) for k in idx))
    return _INDEX


def _match_at(s, pos):
    idx, maxlen = _load()
    for L in range(min(maxlen, len(s) - pos), 1, -1):
        cands = idx.get(s[pos:pos + L])
        if cands:
            return L, cands
    return 0, None


def _prefer_window(cands):
    w = [c for c in cands if c["in_window"]]
    return w or cands


def resolve_place(raw, court_province=None, court_prefecture=None):
    """Return (province, prefecture, county, status).

    status: resolved | province_only | ambiguous_resolved_by_court_prov | ambiguous | unresolved | local_word
    """
    if not isinstance(raw, str):
        return None, None, None, "unresolved"
    s = _PREFIX_JUNK.sub("", raw.strip("：: ，,。"))
    s = re.sub(r"人$", "", s)
    if s.startswith(LOCAL_WORDS):
        if s.startswith("本省"):
            return court_province, None, None, "local_word"
        return court_province, court_prefecture, None, "local_word"
    tokens, pos = [], 0
    while pos < len(s) and len(tokens) < 3:
        L, cands = _match_at(s, pos)
        if not L:
            break
        tokens.append(cands)
        pos += L
        if any(c["level"] == "county" for c in cands) and all(c["level"] == "county" for c in cands):
            break
    if not tokens:
        return None, None, None, "unresolved"
    prov_tok = [c for t in tokens for c in t if c["level"] == "province"]
    provs = {c["province"] for c in prov_tok}
    # the most specific token decides; earlier tokens constrain it
    cands = None
    for t in tokens:
        sub = [c for c in t if c["level"] != "province"]
        if not sub:
            continue
        if provs:
            sub = [c for c in sub if c["province"] in provs] or sub
        if cands:
            prefs = {c["prefecture_code"] for c in cands}
            narrowed = [c for c in sub if c["prefecture_code"] in prefs]
            sub = narrowed or sub
        cands = _prefer_window(sub)
    if not cands:
        if len(provs) == 1:
            return provs.pop(), None, None, "province_only"
        return None, None, None, "unresolved"
    keys = {(c["province"], c["prefecture"]) for c in cands if c["prefecture"]}
    county = None
    cty = {c["full_name"] for c in cands if c["level"] == "county"}
    if len(cty) == 1:
        county = next(iter(cty))
    if len(keys) == 1:
        p, f = next(iter(keys))
        return p, f, county, "resolved"
    if court_province:
        k2 = {k for k in keys if k[0] == court_province}
        if len(k2) == 1:
            p, f = next(iter(k2))
            return p, f, county, "ambiguous_resolved_by_court_prov"
    ps = {k[0] for k in keys}
    return (next(iter(ps)) if len(ps) == 1 else None), None, None, "ambiguous"


@lru_cache(maxsize=None)
def pref_code_of(prefecture_name=None, prefecture_code=None, province=None):
    """2016 prefecture code for a court: from a (possibly old) prefecture code, or a name."""
    idx, _ = _load()
    if province in MUNI_PREFIX.values():
        return {v: k for k, v in MUNI_PREFIX.items()}[province] + "0000"
    try:
        code = str(int(float(prefecture_code))) if prefecture_code is not None else None
    except ValueError:
        code = None
    if code:
        for r in idx.get(prefecture_name, []):
            if r["code"] == code and r["prefecture_code"]:
                return r["prefecture_code"]
    for r in _prefer_window(idx.get(prefecture_name, [])):
        if r["level"] == "prefecture" or (r["level"] == "county" and r["prefecture"] == r["full_name"]):
            return r["prefecture_code"]
    return None


def pref_name_of_code(code):
    idx, _ = _load()
    for recs in idx.values():
        for r in recs:
            if r["prefecture_code"] == code:
                return r["prefecture"]
    return None


@lru_cache(maxsize=None)
def _code_index():
    idx, _ = _load()
    by_code, code2name = {}, {}
    for recs in idx.values():
        for r in recs:
            if r["prefecture_code"]:
                by_code.setdefault(r["code"], r["prefecture_code"])
                code2name.setdefault(r["prefecture_code"], r["prefecture"])
    return by_code, code2name


@lru_cache(maxsize=None)
def court_prefecture(court_city=None, prefecture_code=None, province=None, court_name=None):
    """Standardised (2016) prefecture name of a court.

    Order: municipality -> meta prefecture name/code -> code alone -> parse the court name.
    """
    by_code, code2name = _code_index()
    if province in MUNI_PREFIX.values():
        return province
    city = None if court_city in (None, "unresolved") else court_city
    code = pref_code_of(city, prefecture_code, province) if (city or prefecture_code) else None
    if code is None and prefecture_code is not None:
        try:
            code = by_code.get(str(int(float(prefecture_code))))
        except ValueError:
            code = None
    if code:
        return code2name.get(code)
    if isinstance(court_name, str):
        p, f, _, st = resolve_place(court_name, province)
        if f and st in ("resolved", "ambiguous_resolved_by_court_prov"):
            return f
    return None
