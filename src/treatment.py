"""Treatment assignment for the plea-leniency reform design (direction M).

Dates are constants so that sensitivity analyses can change them in one place.
"""
import re

import numpy as np
import pandas as pd

from src.numparse import cn_to_int

# Pilot: 《关于在部分地区开展刑事案件认罪认罚从宽制度试点工作的办法》 issued 2016-11-16
PILOT_DATE = pd.Timestamp("2016-11-16")
# National: amended Criminal Procedure Law in force 2018-10-26
NATIONAL_DATE = pd.Timestamp("2018-10-26")

PILOT_CITIES = ["北京市", "天津市", "上海市", "重庆市", "沈阳市", "大连市", "南京市", "杭州市",
                "福州市", "厦门市", "济南市", "青岛市", "郑州市", "武汉市", "长沙市", "广州市",
                "深圳市", "西安市"]
MUNICIPALITIES = ["北京市", "天津市", "上海市", "重庆市"]
PROVINCIAL_CAPITALS = ["石家庄市", "太原市", "呼和浩特市", "沈阳市", "长春市", "哈尔滨市", "南京市",
                       "杭州市", "合肥市", "福州市", "南昌市", "济南市", "郑州市", "武汉市", "长沙市",
                       "广州市", "南宁市", "海口市", "成都市", "贵阳市", "昆明市", "拉萨市", "西安市",
                       "兰州市", "西宁市", "银川市", "乌鲁木齐市"]
SUB_PROVINCIAL = ["哈尔滨市", "长春市", "沈阳市", "大连市", "济南市", "青岛市", "南京市", "杭州市",
                  "宁波市", "厦门市", "武汉市", "广州市", "深圳市", "成都市", "西安市"]

_CN_DATE = re.compile(r"([\d〇○零一二三四五六七八九Ｏ]{4})\s*年\s*([\d一二三四五六七八九十]{1,3})\s*月"
                      r"\s*([\d一二三四五六七八九十]{1,3})\s*日")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def _to_int(s):
    return int(s) if s.isdigit() else cn_to_int(s)


def parse_judgment_date(s):
    """Parse '2018-08-28', '2018年8月28日', '2015年十二月二十九日', '二〇一五年一月五日'."""
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    m = _ISO_DATE.match(s) or _CN_DATE.search(s)
    if not m:
        return pd.NaT
    try:
        y, mo, d = (_to_int(g) for g in m.groups())
        return pd.Timestamp(year=y, month=mo, day=d)
    except (TypeError, ValueError):
        return pd.NaT


def date_format_kind(s):
    if not isinstance(s, str) or not s.strip():
        return "missing"
    if _ISO_DATE.match(s):
        return "iso"
    m = _CN_DATE.search(s)
    if m:
        return "cn_arabic" if m.group(2).isdigit() else "cn_numeral"
    return "other"


# "××市××区人民法院" -> "××市"; "××省××市中级人民法院" -> "××市"; autonomous prefectures too
_CITY_FROM_COURT = re.compile(
    r"^(?:[一-龥]{2,8}?(?:省|自治区))?([一-龥]{2,10}?(?:市|自治州|地区|盟))")


def court_city_from_name(court_name, known_prefectures=None):
    """Prefecture-level city from a court name; returns None if only a county-level unit is named."""
    if not isinstance(court_name, str):
        return None
    m = _CITY_FROM_COURT.match(court_name.strip())
    if not m:
        return None
    city = m.group(1)
    if known_prefectures is not None and city not in known_prefectures:
        return None  # e.g. county-level city ("灵宝市") -> leave to T03 gazetteer
    return city


def assign_court_city(meta: pd.DataFrame) -> pd.DataFrame:
    """Return court_city and court_city_source for each row of meta."""
    known = set(meta["prefecture"].dropna().unique())
    city = meta["prefecture"].astype("object").copy()
    src = pd.Series(np.where(city.notna(), "prefecture", None), index=meta.index, dtype="object")
    muni = city.isna() & meta["province"].isin(MUNICIPALITIES)
    city[muni] = meta.loc[muni, "province"]
    src[muni] = "municipality"
    rest = city.isna()
    parsed = meta.loc[rest, "court_name"].map(lambda c: court_city_from_name(c, known))
    ok = parsed.notna()
    city[parsed[ok].index] = parsed[ok]
    src[parsed[ok].index] = "court_name"
    for m in MUNICIPALITIES:  # e.g. "北京市朝阳区人民法院" with province missing
        hit = city.isna() & meta["court_name"].fillna("").str.startswith(m[:2])
        city[hit] = m
        src[hit] = "court_name"
    src[city.isna()] = "unresolved"
    city[city.isna()] = "unresolved"
    return pd.DataFrame({"court_city": city, "court_city_source": src}, index=meta.index)


def months_between(d: pd.Series, start) -> pd.Series:
    """Whole calendar months from start to d (negative before start)."""
    start = pd.Series(start, index=d.index) if not isinstance(start, pd.Series) else start
    out = (d.dt.year - start.dt.year) * 12 + (d.dt.month - start.dt.month)
    out = out - (d.dt.day < start.dt.day).astype("float")
    return out.where(d.notna())


def assign_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """df needs judgment_date_parsed (datetime64) and court_city."""
    d = df["judgment_date_parsed"]
    pilot = df["court_city"].isin(PILOT_CITIES)
    post_pilot = pilot & (d >= PILOT_DATE)
    post_national = d >= NATIONAL_DATE
    start = pd.Series(np.where(pilot, PILOT_DATE, NATIONAL_DATE), index=df.index).astype("datetime64[ns]")
    em = months_between(d, start)
    city_type = np.select(
        [pilot, df["court_city"].isin(PROVINCIAL_CAPITALS + SUB_PROVINCIAL)],
        ["pilot", "capital_or_subprov"], default="other")
    return pd.DataFrame({
        "pilot_city": pilot,
        "city_type": city_type,
        "post_pilot": post_pilot.astype("boolean").mask(d.isna()),
        "post_national": post_national.astype("boolean").mask(d.isna()),
        "post": (post_pilot | post_national).astype("boolean").mask(d.isna()),
        "treat_start": start,
        "event_time_m": em.astype("Int32"),
        "event_time_q": np.floor(em / 3).astype("Int32"),
    }, index=df.index)
