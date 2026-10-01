"""Split a criminal judgment into header / charge / facts / reasoning / verdict.

header    : parties (defendants, counsel) -- from the start up to the procedural sentence
            ("××检察院以××号起诉书指控…") or, if absent, the next segment start
charge    : prosecution's allegation ("公诉机关指控…")
facts     : court's findings ("经审理查明…")
reasoning : "本院认为…"
verdict   : after "判决如下"
Segments that are not found are None. Order is enforced: each marker is searched after
the previous segment's start.
"""
import re

SEGMENTS = ["header", "charge", "facts", "reasoning", "verdict"]

PROCEDURAL = re.compile(
    r"(?:人民检察院|检察院|公诉机关|检察分院)(?:于\d{4}年[^。，]{0,12}日)?(?:以|依)[^。]{0,80}?"
    r"(?:起诉书|起诉决定书|诉字|刑诉)")
CHARGE = re.compile(
    r"(?:公诉机关|人民检察院|检察院|检察分院|公诉人)(?:当庭)?(?:起诉)?指控(?:情况)?(?:称|认为)?[：:，,\s]"
    r"|起诉书(?:中)?指控[：:]"
    r"|起诉书指控(?=[，,]?\s*(?:\d{4}年|[一二三四五六七八九十]+、|同年|自|被告人[^。，]{1,10}(?:于|在|伙同|先后|多次|窜至|趁)))"
    r"|公诉机关指控情况")
FACTS = re.compile(
    r"(?:经|本院经?)(?:本院|法庭|依法|过)?(?:公开)?(?:开庭)?(?:审理|庭审|审查)[，,]?(?:本院)?(?:查明|认定)"
    r"|(?<!另)(?:本院)?审理查明|本院查明(?:事实)?|经庭审(?:查明|认定)|法庭审理查明")
REASONING = re.compile(r"本院认为|本庭认为|判决理由|经本院审理，本案事实清楚")
VERDICT = re.compile(r"判决如下|判决结果|判处如下|判决[：:]"
                     r"|判\ufffd{1,3}如下|判决如\ufffd{1,3}|\ufffd{1,3}决如下")
# fallback when no verdict marker: the first "被告人X犯Y罪，判处/免予…" after the reasoning start
VERDICT_FALLBACK = re.compile(r"被告(?:人|单位)?[^，。；]{1,12}?犯[^，。；]{1,20}?罪[，,](?:判处|免予|免除|单处)")

def find_spans(text: str) -> dict:
    """Return {segment: (start, end) or None}."""
    text = text or ""
    n = len(text)
    marks = {}
    proc = PROCEDURAL.search(text)
    ch = CHARGE.search(text)
    marks["charge"] = ch.start() if ch else None
    after_charge = marks["charge"] or 0
    fa = FACTS.search(text, after_charge)
    marks["facts"] = fa.start() if fa else None
    base = max(p for p in (marks["charge"], marks["facts"], 0) if p is not None)
    re_ = REASONING.search(text, base)
    if not re_:  # reasoning may precede a mis-detected facts marker
        re_ = REASONING.search(text, after_charge)
    marks["reasoning"] = re_.start() if re_ else None
    ve = VERDICT.search(text, marks["reasoning"] or after_charge)
    if ve:
        marks["verdict"] = ve.end()
    else:
        vf = VERDICT_FALLBACK.search(text, marks["reasoning"] or after_charge)
        marks["verdict"] = vf.start() if vf else None
    if marks["facts"] is not None and marks["reasoning"] is not None and marks["facts"] > marks["reasoning"]:
        marks["facts"] = None
    # header end: procedural sentence, else first later segment
    later = [p for p in (marks["charge"], marks["facts"], marks["reasoning"], marks["verdict"]) if p is not None]
    p_start = None
    if proc:  # back up to the sentence boundary so "××市人民检察院以…" is not left in the header
        p_start = proc.start()
        back = max(text.rfind(c, 0, p_start) for c in "。；;")
        if back >= 0 and p_start - back <= 40:
            p_start = back + 1
    cands = ([p_start] if p_start is not None else []) + later
    h_end = min(cands) if cands else None
    spans = {"header": (0, h_end) if h_end and "被告" in text[:h_end] else None}
    order = ["charge", "facts", "reasoning", "verdict"]
    starts = [(k, marks[k]) for k in order if marks[k] is not None]
    for i, (k, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else n
        if k == "verdict":
            e = n
        spans[k] = (s, e) if e > s else None
    for k in order:
        spans.setdefault(k, None)
    return spans


def segment(text: str) -> dict:
    spans = find_spans(text)
    return {k: (text[sp[0]:sp[1]] if sp else None) for k, sp in spans.items()}
