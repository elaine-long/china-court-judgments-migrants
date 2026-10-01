"""Rule-based (regex) extraction of defendant- and document-level variables (T02 baseline).

Entry point: extract_doc(doc_id, text, judgment_date, court_province) -> (doc_row, [def_rows])
"""
import re
from datetime import date

from src.gazetteer import resolve_place
from src.numparse import CN_CHARS, NUM, parse_duration_months, parse_number
from src.segment import PROCEDURAL, find_spans

# ----------------------------------------------------------------------------- places
PROVINCES = {
    "北京市": ["北京"], "天津市": ["天津"], "上海市": ["上海"], "重庆市": ["重庆"],
    "河北省": ["河北"], "山西省": ["山西"], "辽宁省": ["辽宁"], "吉林省": ["吉林省"],
    "黑龙江省": ["黑龙江"], "江苏省": ["江苏"], "浙江省": ["浙江"], "安徽省": ["安徽"],
    "福建省": ["福建"], "江西省": ["江西"], "山东省": ["山东"], "河南省": ["河南"],
    "湖北省": ["湖北"], "湖南省": ["湖南"], "广东省": ["广东"], "海南省": ["海南省"],
    "四川省": ["四川"], "贵州省": ["贵州"], "云南省": ["云南"], "陕西省": ["陕西"],
    "甘肃省": ["甘肃"], "青海省": ["青海"], "台湾省": ["台湾"],
    "内蒙古自治区": ["内蒙古"], "广西壮族自治区": ["广西"], "西藏自治区": ["西藏"],
    "宁夏回族自治区": ["宁夏"], "新疆维吾尔自治区": ["新疆"],
    "香港特别行政区": ["香港"], "澳门特别行政区": ["澳门"],
}
_PROV_PREFIXES = sorted(((p, full) for full, ps in PROVINCES.items() for p in [full] + ps),
                        key=lambda x: -len(x[0]))
LOCAL_WORDS = ("本市", "本区", "本县", "本省", "本镇", "本乡", "本地", "本辖区")

# prefecture -> province, set by set_prefecture_map() (built from meta)
_PREF_MAP: dict = {}
_PREF_KEYS: list = []


def set_prefecture_map(mapping: dict):
    global _PREF_MAP, _PREF_KEYS
    m = {}
    for pref, prov in mapping.items():
        if not pref or not prov:
            continue
        m[pref] = prov
        if pref.endswith("市") and len(pref) >= 3:
            m.setdefault(pref[:-1], prov)
    _PREF_MAP = m
    _PREF_KEYS = sorted(m, key=len, reverse=True)


def province_of(place):
    """Province of a raw place string; 'LOCAL' for 本市/本县…; None if only county-level/unknown."""
    if not isinstance(place, str) or not place:
        return None
    s = place.strip("：: ")
    if s.startswith(LOCAL_WORDS):
        return "LOCAL"
    for key in _PREF_KEYS:  # prefecture first (handles 吉林市, 海南藏族自治州)
        if s.startswith(key):
            nxt = s[len(key):len(key) + 1]
            if not key.endswith(("市", "州", "盟", "地区")) and nxt in ("区", "县", "镇", "乡"):
                continue  # stem like 朝阳 followed by 区 -> ambiguous district name
            return _PREF_MAP[key]
    for p, full in _PROV_PREFIXES:
        if s.startswith(p):
            return full
    return None


# ----------------------------------------------------------------------------- helpers
_WS = re.compile(r"[\s　\xa0]+")
# literal newline escapes left in some scraped texts ("＼n山东省莱州市人民法院＼n＼n刑事判决书")
_ESC_NL = re.compile(r"[＼\\][nrt]")
SENT_SPLIT = re.compile(r"[。；;]")


def normalize(text):
    return _WS.sub("", _ESC_NL.sub("", text or ""))


def sentences(s):
    return [x for x in SENT_SPLIT.split(s or "") if x]


# ----------------------------------------------------------------------------- defendants in header
NAME_CHARS = r"[一-龥A-Za-z0-9·•●・‧×＊*ＸX]"
_DEF_MENTION = re.compile(
    r"被告(?:人|�{1,3})(" + NAME_CHARS + r"{1,12}?)(?:[（(][^）)]{0,40}[）)])?"
    # the name ends at punctuation, a gender word, or a clause that starts the case history
    # ("被告人彭某因本案，…", "被告人甲曾因…", "被告人乙于2014年…")
    r"(?=[，,。；;：:]|男|女|$|因本案|因涉嫌|因犯|因盗窃|曾因|曾于|于\d{4}年)")
# words that cannot be part of a name; single function characters only disallowed after the surname
_BAD_WORD = re.compile(r"犯|的|等|辩护|基本|情况|身份|自愿|供述|共同|到庭|盗窃|抢劫|违法|所得|认罪|无罪")
_BAD_TAIL = re.compile(r"[及系对因于在到与和被经]")
_PARA_STOP = re.compile(r"(?:^|[。；;，,])(?:指定|委托)?(?:辩护人|法定代理人|诉讼代理人|附带民事|诉讼参与人)")


def bad_name(name):
    return bool(_BAD_WORD.search(name) or _BAD_TAIL.search(name[1:]))


# a paragraph starts at the beginning of a sentence (or right after the prosecutor / structured labels)
# "号" covers headers where the defendant follows the case number directly ("…刑初696号被告人陈某")
_PARA_START_BEFORE = re.compile(r"(?:^|[。；;：:）)]|院|序|况|号|\s)$")
_IDENTITY_SOON = re.compile(r"男|女|出生|生于|族")


def find_defendants(header):
    """Return list of (name, paragraph) in header order.

    Same-name defendants (two "王某") are kept as separate paragraphs when the repeated
    mention opens a new identity block; otherwise the repeat is a continuation sentence.
    The paragraph stops at the first counsel / representative line.
    """
    out = []
    for name, block in defendant_blocks(header):
        stop = _PARA_STOP.search(block, 5)
        out.append((name, block[:stop.start() + 1] if stop else block))
    return out


def defendant_blocks(header):
    """Return list of (name, block): the raw header text from a defendant's paragraph start up to
    the next defendant's paragraph start (so it includes the counsel lines that follow it)."""
    hits, seen = [], set()
    # try every "被告人" position: a rejected long match ("被告人基本情况被告人杜某") must not
    # swallow the real mention nested inside it
    for start in (s.start() for s in re.finditer(r"被告(?:人|�)", header)):
        m = _DEF_MENTION.match(header, start)
        if not m:
            continue
        name = m.group(1)
        if bad_name(name):
            continue
        if len(name) > 5 and not re.search(r"[·•●・‧A-Za-z]", name):
            continue  # "钱某身患疾病为由拒收" is not a name
        if not _PARA_START_BEFORE.search(header[max(0, m.start() - 1):m.start()]):
            continue
        if name in seen and not _IDENTITY_SOON.search(header[m.end():m.end() + 30]):
            continue
        seen.add(name)
        hits.append((name, m.start()))
    return [(name, header[s:(hits[i + 1][1] if i + 1 < len(hits) else len(header))])
            for i, (name, s) in enumerate(hits)]


_COUNSEL = re.compile(r"(?:指定|委托)?辩护人|辩护律师|法律援助律师|值班律师")
_NO_COUNSEL = re.compile(r"辩护人[：:]?(?:无|没有)(?![^，,。；;]{0,4}(?:律师|事务所))|无辩护人|未委托辩护人|没有委托辩护人")


def counsel_for(name, block, pre, names):
    """Does this defendant have counsel? T06a2 attribution rule.

    1. A counsel line inside the defendant's own block (after its paragraph, before the next
       defendant's paragraph) belongs to that defendant; "辩护人：无" means none.
    2. Otherwise a sentence anywhere before the charge that names the defendant with counsel
       ("被告人X及其辩护人", "被告人X的辩护人", "为被告人X辩护") counts.
    3. Single defendant: any counsel line before the charge counts.
    """
    if _NO_COUNSEL.search(block):
        return False
    if _COUNSEL.search(block[5:]):
        return True
    if names.count(name) == 1:
        n = re.escape(name)
        if re.search(rf"被告人{n}(?:及其|的)(?:指定|委托)?辩护人|为被告人{n}(?:的)?(?:指定)?辩护|被告人{n}[^。；;]{{0,12}}委托[^。；;]{{0,20}}辩护", pre):
            return True
    if len(names) == 1:
        return bool(_COUNSEL.search(pre)) and not _NO_COUNSEL.search(pre)
    return False


# ----------------------------------------------------------------------------- identity fields
_GENDER = re.compile(r"[，,。：:）)]\s*(男|女)(?=[，,。；;、]|性)")
_BIRTH = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(?:出生|生|出世)")
_BIRTH2 = re.compile(r"(?:出生于|生于|出生日期[：:]?)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_ETHNIC = re.compile(r"[，,。]([一-龥]{1,6}?族)(?=[，,。；;])")
_EDU = re.compile(r"[，,。]((?:文盲|半文盲)|(?:小学|初中|初级中学|高中|中专|中技|职高|技校|大专|专科|大学专科|大学本科|大学|本科|研究生|硕士|博士)"
                  r"(?:文化(?:程度)?|毕业|肄业|学历|在读|文化水平)?)(?=[，,。；;])")
_OCC_KW = re.compile(r"[，,。](无业|无固定职业|无职业|农民|务农|务工|农民工|个体(?:户|经营|工商户)?|工人|学生|职员|"
                     r"居民|城镇居民|司机|驾驶员|自由职业|待业|打工|无固定工作|无正当职业|个体劳动者|[^，,。；;]{1,10}(?:员工|员|工|师))"
                     r"(?=[，,。；;])")
_HUKOU = re.compile(r"户(?:籍|口)(?:所在)?(?:地|登记地)?(?:为|在|是|系|于)?[：:]?(?!证明|信息|资料|材料|登记表)([^，,。；;（(]{2,40})")
_BIRTHPLACE = re.compile(r"(?:出生于|生于|出生地[：:为]?)(?!\d)([^，,。；;（(\d]{2,30})")
_NATIVE = re.compile(r"[，,。]([一-龥]{2,20}?(?:省|市|县|区|州|旗|盟)[一-龥]{0,10}?人)(?=[，,。；;])")
# T06a2: also "捕前住址", "户籍地及现住址", "户籍所在地及住址", "同现住址"
_RESID = re.compile(r"(?:^|[，,。；;：:）)]|及|同|和|与)(?:捕前|现|家|暂|租|实际|经常)?(?:居)?住(?:所地|址|所|地)?(?:于|在|为)?[：:]?(?!院|宿)"
                    r"([^，,。；;（(]{2,40})")
_RESID_BAD = re.compile(r"^(?:院|宿|房|所|处)|看守所")


def _first(rx, s, group=1):
    m = rx.search(s)
    return m.group(group) if m else None


def parse_identity(para, judgment_date):
    out = {}
    out["gender"] = _first(_GENDER, para)
    m = _BIRTH.search(para) or _BIRTH2.search(para)
    bd = None
    if m:
        try:
            bd = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            bd = None
    out["birth_date"] = bd
    age = None
    if bd and judgment_date is not None:
        jd = judgment_date
        age = jd.year - bd.year - ((jd.month, jd.day) < (bd.month, bd.day))
        if not 10 <= age <= 100:
            age = None
    out["age_at_judgment"] = age
    out["ethnicity"] = _first(_ETHNIC, para)
    em = _EDU.search(para)
    out["education"] = em.group(1) if em else None
    occ = None
    if em:
        tok = re.match(r"[，,]([^，,。；;]{1,14})(?=[，,。；;])", para[em.end():])
        if tok and not re.match(r"(?:现|家|暂|租)?住|户|身份|因|曾|系|捕前|出生|原", tok.group(1)) \
                and not re.search(r"(?:省|市|县|区|州|旗|盟)[一-龥]{0,10}人$|族$", tok.group(1)):
            occ = tok.group(1)
    if occ is None:
        occ = _first(_OCC_KW, para)
    if occ is None:
        occ = _first(re.compile(r"捕前(?:系|为|是)?([^，,。；;]{1,14})"), para)
    out["occupation_raw"] = occ
    hk = _first(_HUKOU, para)
    if hk:
        hk = re.sub(r"^(?:址|均为|均系|均在)?(?:同|即|及|和|与)?(?:捕前)?(?:现?住址|住所地|居住地|现住|住)?(?:为|在|于)?[：:]?", "", hk)
        if not hk or re.match(r"同上|不详|不明|无$", hk):
            hk = None
    out["hukou_raw"] = hk
    out["birthplace_raw"] = _first(_BIRTHPLACE, para)
    out["native_raw"] = _first(_NATIVE, para)
    res = None
    for rm in _RESID.finditer(para):
        cand = rm.group(1)
        if not _RESID_BAD.search(cand):
            res = cand
            break
    out["residence_raw"] = res
    return out


# ----------------------------------------------------------------------------- prior record & pretrial measures
_CURRENT = re.compile(r"涉嫌|因本案|本案|因[^。，]{0,8}嫌疑|因此次|现因|因该案")
_PRIOR = re.compile(r"判处|刑满释放|假释|有期徒刑|拘役|管制|缓刑|前科|服刑")
_MEAS_ANY = re.compile(r"刑事拘留|刑拘|逮捕|取保候审|监视居住|羁押|关押|抓获|投案")
_NO_PRIOR = re.compile(r"无前科|无犯罪记录|无违法犯罪")

_DATE_TOK = re.compile(
    r"(\d{4})年(\d{1,2})月(\d{1,2})日|同年(\d{1,2})月(\d{1,2})日|同月(\d{1,2})日|(次日|同日|当日|当天|翌日)"
    r"|(?<![年\d])(\d{1,2})月(\d{1,2})日")
MEASURES = {
    "detained": re.compile(r"刑事拘留|刑拘"),
    "arrested": re.compile(r"(?<!不予)(?<!不批准)(?<!未)逮捕"),
    "bail": re.compile(r"取保候审|取报候审|取保"),
    "rsl": re.compile(r"监视居住"),
}
_NOW_CUSTODY = re.compile(r"现(?:被)?(?:羁押|押|在押|关押|羁|寄押)|现在押|现在[^。，]{0,15}服刑|正在服刑|现(?:被)?关押在")
_NOW_BAIL = re.compile(r"现(?:被|已)?取保|取保候审于|取保候审在家|现在家")
_NOW_RSL = re.compile(r"现(?:被)?(?:指定居所)?监视居住|监视居住于")
_NOW_LIVES = re.compile(r"现(?:居)?住|现在[^。，]{0,8}(?:居住|工作)")


def _resolve_dates(s):
    """List of (pos, date) for date mentions in s, resolving 同年/同月/次日 relative to the last date."""
    out, last = [], None
    for m in _DATE_TOK.finditer(s):
        d = None
        try:
            if m.group(1):
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif m.group(4) and last:
                d = date(last.year, int(m.group(4)), int(m.group(5)))
            elif m.group(6) and last:
                d = date(last.year, last.month, int(m.group(6)))
            elif m.group(7) and last:
                d = last if m.group(7) in ("同日", "当日", "当天") else date.fromordinal(last.toordinal() + 1)
            elif m.group(8) and last:
                d = date(last.year, int(m.group(8)), int(m.group(9)))
        except ValueError:
            d = None
        if d:
            out.append((m.start(), d))
            last = d
    return out


def parse_prior_and_measures(para, extra=""):
    """para: defendant paragraph; extra: procedural sentences about this defendant."""
    out = {}
    # the current case starts at "涉嫌/因本案…" or at the first measure sentence that is not a prior conviction
    cm = _CURRENT.search(para)
    cur = cm.start() if cm else len(para)
    pos = 0
    for sent in re.split(r"(?<=[。；;])", para):
        if pos >= cur:
            break
        if _MEAS_ANY.search(sent) and not _PRIOR.search(sent):
            cur = pos
            break
        pos += len(sent)
    before = para[:cur]
    prior_sents = [x for x in re.split(r"[。；;]", before) if _PRIOR.search(x)]
    out["prior_record"] = bool(prior_sents) and not _NO_PRIOR.search(before)
    out["prior_theft"] = any("盗窃" in x for x in prior_sents)
    region = para[cur:] + ("。" + extra if extra else "")
    dates = _resolve_dates(region)
    events = []
    for k, rx in MEASURES.items():
        first = None
        for m in rx.finditer(region):
            events.append((m.start(), k))
            if first is None:
                d = [dd for p, dd in dates if p < m.start() and m.start() - p <= 60]
                first = d[-1] if d else None
                out[k] = True
                out[f"{k}_date"] = first
        out.setdefault(k, False)
        out.setdefault(f"{k}_date", None)
    out["pretrial_status_at_judgment"] = status_at_judgment(region, events)
    return out


def status_at_judgment(region, events):
    """在押 / 取保 / rsl. A "现…" state sentence wins; otherwise the last measure event."""
    had_bail = any(k == "bail" for _, k in events)
    for sent in reversed([x for x in re.split(r"[。；;]", region) if x]):
        # within one sentence the last-mentioned state counts ("…逮捕，后…拒收，现取保候审于…")
        found = []
        for rx, lab in ((_NOW_CUSTODY, "在押"), (_NOW_BAIL, "取保"), (_NOW_RSL, "rsl")):
            found += [(m.start(), lab) for m in rx.finditer(sent)]
        if not found and had_bail and _NOW_LIVES.search(sent):
            found = [(0, "取保")]
        if found:
            return max(found)[1]
    if events:
        last = max(events)[1]
        return {"detained": "在押", "arrested": "在押", "bail": "取保", "rsl": "rsl"}[last]
    return None


# ----------------------------------------------------------------------------- verdict: charges & penalty
_CHARGE = re.compile(r"犯([^罪，,。；;：:、（(\d]{1,15}?罪)")
_CHARGE_CONT = re.compile(r"罪[、和及]([^罪，,。；;：:、（(\d]{1,15}?罪)")
_BAD_CHARGE = re.compile(r"^(?:罪|犯|被告|该|其|上述|本|前|所|又|数|新)")
_PEN = re.compile(r"(死刑|无期徒刑|有期徒刑|拘役|管制)([" + CN_CHARS + r"\d年个月零日天]*)")
_PROB = re.compile(r"缓刑([" + CN_CHARS + r"\d年个月零日天]+)")
_FINE = re.compile(r"罚金(?:人民币)?(" + NUM + r")(?:余)?元")
_PAREN = re.compile(r"[（(][^（）()]*[）)]")
_VERDICT_END = re.compile(r"如不服本判决|如不服本裁判|审判长|审判员|人民陪审员")


def parse_charges(chunk):
    out = []
    for rx in (_CHARGE, _CHARGE_CONT):
        for m in rx.finditer(chunk):
            c = m.group(1)
            if not _BAD_CHARGE.search(c) and c not in out:
                out.append(c)
    return out


def parse_penalty(chunk):
    out = {"penalty_type": None, "term_months": None, "probation": False, "probation_months": None,
           "fine_yuan": None}
    c = _PAREN.sub("", chunk)
    i = c.rfind("决定执行")
    sub = c[i:] if i >= 0 else c[c.find("判处") if "判处" in c else 0:]
    m = _PEN.search(sub)
    if m:
        kind = m.group(1)
        out["penalty_type"] = kind
        if kind in ("有期徒刑", "拘役", "管制"):
            out["term_months"] = parse_duration_months(m.group(2))
        if kind == "死刑" and "缓期" in sub[m.end():m.end() + 6]:
            out["penalty_type"] = "死缓"
        pm = _PROB.search(sub, m.end())
        if pm:
            out["probation"] = True
            out["probation_months"] = parse_duration_months(pm.group(1))
    elif re.search(r"免予刑事处罚|免除刑事处罚|免于刑事处罚", sub):
        out["penalty_type"] = "免予刑事处罚"
    elif re.search(r"单处罚金|罚金", sub):
        out["penalty_type"] = "单处罚金"
    fm = _FINE.search(sub) or _FINE.search(c)
    if fm:
        out["fine_yuan"] = parse_number(fm.group(1))
    return out


def split_verdict(verdict, names):
    """Verdict chunk for each defendant (list aligned with names; None if not found).

    The k-th defendant called "王某" gets the k-th verdict item "被告人王某犯…".
    """
    v = verdict
    e = _VERDICT_END.search(v)
    if e:
        v = v[:e.start()]
    starts = [None] * len(names)
    used = set()
    for i, n in enumerate(names):
        # lookahead keeps 李某 from matching 李某某 / 李某甲
        base = r"被告(?:人|�{1,3})?" + re.escape(n) + r"(?![某甲乙丙丁戊\dA-Za-z])"
        items = [m.start() for m in re.finditer(base + r"(?=犯|无罪|免予)", v)]
        if not items and names.count(n) == 1:
            m = re.search(base, v)
            items = [m.start()] if m else []
        for p in items:
            if p not in used:
                starts[i] = p
                used.add(p)
                break
    order = sorted(p for p in starts if p is not None)
    chunks = []
    for p in starts:
        if p is None:
            chunks.append(None)
            continue
        nxt = [q for q in order if q > p]
        chunks.append(v[p:nxt[0] if nxt else len(v)])
    return chunks, v


def count_verdict_items(vclean):
    return len({m.start() for m in _VERDICT_DEF.finditer(vclean)})


_VERDICT_DEF = re.compile(r"被告(?:人|�{1,3})(" + NAME_CHARS + r"{1,12}?)(?=犯[^罪，。]{1,15}罪|无罪)")

# ----------------------------------------------------------------------------- amounts
AMT_EXCL = re.compile(r"退赔|退赃|退还|退回|赔偿|罚金|追回|追缴|发还|返还|获利|销赃|得款|获款|卖得|售得|出售|变卖|"
                      r"典当|收购|卖给|抵押|谅解|担保|保证金|违法所得|挽回|赃款已|损失已|补偿|医疗|修复|维修|购买|购得|"
                      r"价格卖|以[^，。]{0,10}价格|退缴|缴纳|借款|工资|欠款|报酬|充值")
_AMT_TOTAL = re.compile(r"(?:共计|合计|总计|累计|总价值|总额|总金额|共)(?:价值|金额|数额)?(?:为|达|计)?(?:人民币)?(" + NUM + r")(?:余|多)?元")
_AMT_ITEM = re.compile(r"(?:价值|数额|金额|价格(?:为|认定为|鉴定为)|认定价格为|估价为|鉴定价值为)(?:共计|合计)?(?:为|达|计|约)?(?:人民币)?(" + NUM + r")(?:余|多)?元")
_AMT_CASH = re.compile(r"(?:盗窃|窃取|窃得|盗走|盗得|扒窃|偷走|偷得|偷取|抢劫|抢得|劫得|劫取|抢走|抢夺|转走|盗刷)[^，。；]{0,20}?"
                       r"(?:现金|人民币|钱款|款)(?:人民币)?(?:\(下同\)|（下同）)?(" + NUM + r")(?:余|多)?元")


_CLAUSE_SPLIT = re.compile(r"[。；;：:（）()]|(?<!\d)[，,]|[，,](?!\d{3})")


# "窃得收银机内800余元": obtaining verb followed directly by an amount (no 价值/现金 word)
_AMT_GOT = re.compile(r"(?:窃得|盗得|偷得|抢得|劫得|骗得|扒得)[^，。；价值]{0,12}?(?<![\d.])(" + NUM + r")(?:余|多)?元(?!的)")


def _amount_matches(seg):
    """Amounts in clauses that do not mention recovery/compensation/fines etc."""
    tot, items = [], []
    for cl in _CLAUSE_SPLIT.split(seg):
        if AMT_EXCL.search(cl):
            continue
        for m in _AMT_TOTAL.finditer(cl):
            v = parse_number(m.group(1))
            if v:
                tot.append(v)
        for rx in (_AMT_ITEM, _AMT_CASH, _AMT_GOT):
            for m in rx.finditer(cl):
                v = parse_number(m.group(1))
                if v:
                    items.append(v)
    return tot, items


def extract_amount(reasoning, facts, charge):
    for name, seg in (("reasoning", reasoning), ("facts", facts), ("charge", charge)):
        if not seg:
            continue
        tot, items = _amount_matches(seg)
        if tot:
            return max(tot), f"{name}_total"
        if items:
            uniq = sorted(set(items))
            if name == "reasoning" and len(uniq) == 1:
                return uniq[0], "reasoning"
            return float(sum(uniq)), f"{name}_sum"
    return None, None


# ----------------------------------------------------------------------------- doc-level flags
def _has_pos(seg, rx, neg=None):
    for s in sentences(seg):
        if rx.search(s) and not (neg and neg.search(s)):
            return True
    return False


_SURR = re.compile(r"自首")
_SURR_NEG = re.compile(r"不(?:构成|能认定|认定|属于|成立|具有|系|予认定|符合|应认定|能成立)[^，。]{0,6}自首|自首[^，。]{0,8}不(?:能|予|成立|构成|符合)|不具有自首|非自首")
_FORG = re.compile(r"谅解")
_FORG_NEG = re.compile(r"未(?:能)?(?:取得|获得|得到)[^，。]{0,6}谅解|没有(?:取得|获得|得到)[^，。]{0,6}谅解|不予谅解|不谅解")
_REST = re.compile(r"退赃|退赔|退还[^，。]{0,6}(?:赃|款|财物|被害人)|赔偿[^，。]{0,8}损失|已赔偿|积极赔偿")
_RECID_NEG = re.compile(r"不(?:构成|属于|系|是|应认定为?|能认定为?)累犯")
_DENIAL = re.compile(r"社区矫正|不宜宣告缓刑|不宜适用缓刑|不适用缓刑|不予适用缓刑|不宜判处缓刑|不予宣告缓刑|无固定住所|不具备[^。；]{0,20}条件")


OUTLIER_YUAN = 1e7
# speedy-trial procedure: explicit application, or the 速 marker in the case number (城刑（速）初字, 刑速)
_SPEEDY = re.compile(r"(?<!不)(?<!未)适用(?:刑事案件)?速裁程序|速裁程序(?:审理|进行审理|公开开庭)|[（(]速[）)]|刑速")

# T06a2: procedure only from an explicit statement ("适用……程序"), as codebook v2 requires; the last
# explicit statement wins ("不宜适用简易程序……决定适用普通程序" -> 普通). The collegial panel is recorded
# separately as panel_trial and no longer implies 普通.
_PROC_EXPLICIT = re.compile(
    r"(?:(?<!不)(?<!不宜)(?<!未)(?<!不能)适用|转为(?:适用)?|转换为|变更为|改为|适用程序[：:]?)(?:刑事案件)?(速裁|简易|普通)程序")
_SPEEDY_CASE_NO = re.compile(r"[（(]速[）)]|刑速")


def explicit_procedure(text):
    hits = _PROC_EXPLICIT.findall(text)
    if hits:
        return hits[-1]
    if _SPEEDY_CASE_NO.search(text[:200]):  # speedy marker in the case number
        return "速裁"
    return None


ORIGIN_ORDER = ("hukou_raw", "birthplace_raw", "native_raw", "residence_raw")


def _migrant(prov, pref, court_prov, court_pref, level):
    """True/False/None. City level: different province implies different city."""
    if prov is None or not court_prov:
        return None
    if prov != court_prov:
        return True
    if level == "prov":
        return False
    if pref is None or court_pref is None:
        return None
    return pref != court_pref


def origin_fields(row, court_province, court_prefecture):
    """Origin by priority hukou > birthplace > native place > residence, resolved with the gazetteer."""
    res = {k: resolve_place(row.get(k), court_province, court_prefecture) if row.get(k) else None
           for k in ORIGIN_ORDER}
    out = {}
    src = next((k for k in ORIGIN_ORDER if row.get(k)), None)
    out["origin_source"] = src[:-4] if src else None
    out["origin_raw"] = row.get(src) if src else None
    prov, pref, county, status = res[src] if src else (None, None, None, None)
    out.update(origin_province_g=prov, origin_prefecture=pref, origin_county=county, origin_status=status)
    out["migrant_city"] = _migrant(prov, pref, court_province, court_prefecture, "city")
    out["migrant_prov"] = _migrant(prov, pref, court_province, court_prefecture, "prov")
    src_nr = next((k for k in ORIGIN_ORDER[:3] if row.get(k)), None)
    if src_nr:
        p2, f2, _, _ = res[src_nr]
        out["migrant_city_nores"] = _migrant(p2, f2, court_province, court_prefecture, "city")
    else:
        out["migrant_city_nores"] = None
    if res["residence_raw"]:
        p3, f3, _, _ = res["residence_raw"]
        mig = _migrant(p3, f3, court_province, court_prefecture, "city")
        out["local_residence"] = None if mig is None else not mig
    else:
        out["local_residence"] = None
    return out


def extract_doc(doc_id, text, judgment_date=None, court_province=None, court_prefecture=None):
    t = normalize(text)
    sp = find_spans(t)
    seg = {k: (t[v[0]:v[1]] if v else None) for k, v in sp.items()}
    header = seg["header"] or ""
    # procedural block: from the header end to the next segment start
    h_end = sp["header"][1] if sp["header"] else 0
    nxt = [v[0] for k, v in sp.items() if v and k != "header" and v[0] >= h_end]
    proc = t[h_end:min(nxt)] if nxt else ""
    reasoning, facts, charge, verdict = seg["reasoning"] or "", seg["facts"] or "", seg["charge"] or "", seg["verdict"] or ""

    defs = find_defendants(header)
    names = [n for n, _ in defs]
    vchunks, vclean = split_verdict(verdict, names) if verdict else ([None] * len(names), "")
    # no defendant paragraph in the header: fall back to the names convicted in the verdict
    if not defs:
        for m in _VERDICT_DEF.finditer(vclean):
            n = m.group(1)
            if n not in names and not bad_name(n):
                names.append(n)
                defs.append((n, ""))
        vchunks, vclean = split_verdict(verdict, names) if verdict else ([None] * len(names), "")
    n_def = len(defs)
    n_items = count_verdict_items(vclean) if vclean else 0
    dup_pseudonym = len(set(names)) < len(names) or (n_items > 0 and n_items != n_def)
    pre = header + proc
    blocks = [b for _, b in defendant_blocks(header)] if header else []

    def_rows = []
    for i, (name, para) in enumerate(defs):
        row = {"doc_id": doc_id, "def_idx": i, "def_name": name, "def_header_raw": para[:400],
               "dup_pseudonym": dup_pseudonym, "name_repeated": names.count(name) > 1}
        row.update(parse_identity(para, judgment_date))
        extra = "。".join(s for s in sentences(proc) if (name in s and names.count(name) == 1) or n_def == 1)
        row.update(parse_prior_and_measures(para, extra))
        prov = province_of(row["hukou_raw"])
        src = "hukou" if prov else None
        if prov is None:
            prov = province_of(row["birthplace_raw"])
            src = "birthplace" if prov else None
        row["origin_province"] = court_province if prov == "LOCAL" else prov
        row["migrant_src"] = src
        row["migrant_v0"] = (None if (row["origin_province"] is None or not court_province)
                             else row["origin_province"] != court_province)
        row.update(origin_fields(row, court_province, court_prefecture))
        row["counsel"] = counsel_for(name, blocks[i] if i < len(blocks) else "", pre, names)
        rs = [s for s in sentences(reasoning) if "累犯" in s and (name in s or n_def == 1)]
        row["recidivist"] = any(not _RECID_NEG.search(s) for s in rs)
        chunk = vchunks[i] if i < len(vchunks) else None
        if chunk is None and n_def == 1:
            chunk = vclean
        chunk = chunk or ""
        row["charges"] = parse_charges(chunk)
        row.update(parse_penalty(chunk) if chunk else
                   {"penalty_type": None, "term_months": None, "probation": None,
                    "probation_months": None, "fine_yuan": None})
        row["fine_outlier"] = bool(row["fine_yuan"] is not None and row["fine_yuan"] > OUTLIER_YUAN)
        row["fine_yuan_ctrl"] = None if row["fine_outlier"] else row["fine_yuan"]
        row["outcomes_ok"] = row["penalty_type"] is not None and (
            row["penalty_type"] not in ("有期徒刑", "拘役") or row["term_months"] is not None)
        def_rows.append(row)

    charges_all = []
    for r in def_rows:
        for c in r["charges"]:
            if c not in charges_all:
                charges_all.append(c)
    if not def_rows and vclean:
        charges_all = parse_charges(vclean)
    amt, amt_src = extract_amount(reasoning, facts, charge)
    body = (facts or charge) + reasoning
    before_verdict = t[:sp["verdict"][0]] if sp["verdict"] else t
    procedure = explicit_procedure(before_verdict)
    speedy = procedure == "速裁" or bool(_SPEEDY.search(t))
    counsel_m = re.search(r"(指定|委托)?辩护人|辩护律师|法律援助律师", pre)
    counsel_type = None
    if counsel_m:
        counsel_type = "指定" if (counsel_m.group(1) == "指定" or re.search(r"法律援助|指定辩护", pre)) else "委托"
    doc = {
        "doc_id": doc_id,
        "has_fffd": "�" in t,
        **{f"has_{k}": v is not None for k, v in sp.items()},
        **{f"len_{k}": (v[1] - v[0]) if v else 0 for k, v in sp.items()},
        "n_defendants": n_def,
        "charges_all": charges_all,
        "is_theft": any("盗窃" in c for c in charges_all),
        "is_robbery": any(c == "抢劫罪" for c in charges_all),
        "charge_main": (def_rows[0]["charges"][0] if def_rows and def_rows[0]["charges"]
                        else (charges_all[0] if charges_all else None)),
        "plea_leniency": bool(re.search(r"认罪认罚|具结书", t)),  # T02 definition, kept for comparison
        "plea_formal": "认罪认罚" in t,
        "jiejie": "具结" in t,
        "speedy": speedy,
        "leniency_proc": ("认罪认罚" in t) or ("具结" in t) or speedy,
        "procedure": procedure,
        "panel_trial": "组成合议庭" in pre,
        "dup_pseudonym": dup_pseudonym,
        "n_verdict_items": n_items,
        "counsel": bool(counsel_m),
        "counsel_type": counsel_type,
        "duty_lawyer": "值班律师" in t,
        "amt_yuan": amt,
        "amt_source": amt_src,
        "amt_outlier": bool(amt is not None and amt > OUTLIER_YUAN),
        "amt_yuan_ctrl": None if (amt is not None and amt > OUTLIER_YUAN) else amt,
        "spec_burglary": bool(re.search(r"入户|入室盗窃|入室行窃", body)),
        "spec_pickpocket": "扒窃" in body,
        "spec_multiple": bool(re.search(r"多次", body)),
        "spec_weapon": bool(re.search(r"携带凶器|持刀|持械|携带刀具|持凶器", body)),
        "attempted": "未遂" in body,
        "surrender": _has_pos(reasoning or t, _SURR, _SURR_NEG),
        "confession": bool(re.search(r"坦白|如实供述", reasoning or t)),
        "restitution": bool(_REST.search(reasoning or "")),
        "forgiveness": _has_pos(reasoning or t, _FORG, _FORG_NEG),
        "accomplice": n_def > 1 or bool(re.search(r"伙同|共同犯罪|结伙|另案处理|另案", body)),
        "probation_denial_text": " || ".join(s for s in sentences(reasoning) if _DENIAL.search(s)) or None,
    }
    return doc, def_rows
