"""Number parsing for Chinese judgment text: Arabic (with thousands separators,
full-width digits/commas, decimals), Chinese numerals, and 万/亿 units."""
import re

_FW = str.maketrans("０１２３４５６７８９．，", "0123456789.,")
_CN_DIGIT = {"零": 0, "〇": 0, "○": 0, "Ｏ": 0, "O": 0, "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2,
             "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5, "六": 6, "陆": 6,
             "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9}
_CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CN_BIG = {"万": 10_000, "萬": 10_000, "亿": 100_000_000}

CN_CHARS = "零〇○一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬"
# A number token: Arabic with separators/decimals, optionally followed by 万/亿, or Chinese numerals.
NUM = (r"(?:\d[\d,，]*(?:\.\d+)?(?:\s*[万亿])?(?:[" + CN_CHARS + r"]*)"
       r"|[" + CN_CHARS + r"]+)")
NUM_RE = re.compile(NUM)


def cn_to_int(s: str):
    """Convert a pure Chinese numeral string to int ('三千五百' -> 3500, '二〇一五' -> 2015)."""
    if not s:
        return None
    if all(ch in _CN_DIGIT for ch in s):  # digit-by-digit, e.g. 二〇一五
        return int("".join(str(_CN_DIGIT[ch]) for ch in s))
    total, section, digit = 0, 0, None
    for ch in s:
        if ch in _CN_DIGIT:
            digit = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            section += (1 if digit is None else digit) * _CN_UNIT[ch]
            digit = None
        elif ch in _CN_BIG:
            section += digit or 0
            total = (total + section) * _CN_BIG[ch] if ch == "亿" else total + section * _CN_BIG[ch]
            section, digit = 0, None
        else:
            return None
    return total + section + (digit or 0)


def parse_number(s):
    """Parse a numeric token to float. Returns None if unparseable.

    Examples: '1,996' -> 1996, '3，000' -> 3000, '6.2万' -> 62000, '三千五百' -> 3500,
    '1万2千' -> 12000, '24332.30' -> 24332.3
    """
    if s is None:
        return None
    s = str(s).translate(_FW).strip().rstrip("余多")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    if re.fullmatch(r"[" + CN_CHARS + r"]+", s):
        v = cn_to_int(s)
        return float(v) if v is not None else None
    m = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)([万亿]?)([\d" + CN_CHARS + r"]*)", s)
    if not m:
        return None
    head = m.group(1)
    # a comma followed by exactly 3 digits is a thousands separator; otherwise treat as decimal-ish noise
    if "," in head:
        if not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", head):
            head = head.split(",")[0]
        head = head.replace(",", "")
    v = float(head)
    unit, tail = m.group(2), m.group(3)
    if unit:
        v *= _CN_BIG[unit]
        if tail:  # e.g. 1万2千
            t = parse_number(tail)
            if t is not None:
                v += t
    elif tail:
        # e.g. '3千' or '5百'
        if tail in ("千", "仟"):
            v *= 1000
        elif tail in ("百", "佰"):
            v *= 100
        elif tail == "十":
            v *= 10
    return v


def parse_duration_months(s):
    """'一年六个月' -> 18, '八个月' -> 8, '十年' -> 120, '十五日' -> 0.5, '1年零3个月' -> 15."""
    if not s:
        return None
    s = s.translate(_FW)
    total, hit = 0.0, False
    m = re.search(r"([\d" + CN_CHARS + r"]+)年", s)
    if m:
        v = parse_number(m.group(1))
        if v is not None:
            total += 12 * v
            hit = True
    m = re.search(r"(?:年零?|^)([\d" + CN_CHARS + r"]+)个?月", s)
    if m:
        v = parse_number(m.group(1))
        if v is not None:
            total += v
            hit = True
    m = re.search(r"(?:月零?|^)([\d" + CN_CHARS + r"]+)[日天]", s)
    if m:
        v = parse_number(m.group(1))
        if v is not None:
            total += v / 30
            hit = True
    return round(total, 3) if hit else None
