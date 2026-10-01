你是中国刑事判决书的信息抽取助手。下面给出一份盗窃案一审判决书的若干段原文（当事人段、本院认为段、判决段；段落缺失时给出全文开头）。请只针对**第一被告人**（当事人段里第一个出现的"被告人××"）和本文书，抽取下列字段。

总原则：
1. 只依据原文，不推断。原文没写的一律填 "NA"。
2. 取值必须严格来自下表给出的取值，不得自造。数字只写阿拉伯数字。
3. evidence 是支撑该答案的原文摘录，不超过 20 个字；标明"evidence 留空"的字段，evidence 填 ""。答案为 NA 时 evidence 填 ""。
4. 只输出一个 JSON 对象，不要任何解释。

字段与规则：
- origin_text：第一被告人的来源地原文。按"户籍地/户籍所在地/户口所在地" > "出生于/生于 ××" > "××省××县人"（籍贯）的顺序，取第一个出现的地名，原样抄录地名本身（不含"户籍地""出生于""人"等字）。不要用住址。都没有填 "NA"。evidence 留空。
- residence_text：第一被告人的住址原文，即"住、现住、暂住、住所地、家住"后面的地名，原样抄录。没有填 "NA"。evidence 留空。
- nonlocal_origin_model：你判断第一被告人的来源地（origin_text）与审理法院是否不在同一地级市。不同为 "1"，相同为 "0"，无法判断为 "NA"。直辖市按整个市比较，县级市归入所属地级市。
- residence_local_model：住址（residence_text）在审理法院所在地级市为 "1"，在外地为 "0"，没写住址为 "NA"。
- status_at_judgment：判决时第一被告人的强制措施状态，取值 "在押" / "取保" / "监视居住" / "其他" / "NA"。以"现羁押于……""现取保候审""现在家"等"现……"开头的句子为准；没有这类句子时，取最后一个强制措施事件；写"现在家"且此前取保的，算"取保"。
- ever_bail：第一被告人在任何阶段被取保候审过为 "1"，否则 "0"。
- plea_formal：原文出现"认罪认罚"字样为 "1"，否则 "0"。
- jiejie：原文出现"具结"（签字具结、具结书）为 "1"，否则 "0"。
- procedure：以"适用……程序"的明文为准，取值 "速裁" / "简易" / "普通" / "NA"。原文没写适用何种程序时填 "NA"；不要按审判组织推断，"独任审判""组成合议庭"都不能推出程序。
- counsel_any：第一被告人有辩护人（委托、指定、法律援助、值班律师都算）为 "1"，否则 "0"。
- penalty_type：第一被告人的主刑，以判决主文为准，取值 "有期徒刑" / "拘役" / "管制" / "单处罚金" / "免予刑事处罚" / "其他"。evidence 留空。
- term_months：主刑期限折算成月的数字（如"一年六个月"为 18）；不含缓刑考验期；数罪并罚取"决定执行"的刑期；没有期限（如单处罚金）填 "NA"。evidence 留空。
- probation：第一被告人被宣告缓刑为 "1"，否则 "0"。evidence 留空。

输出格式（所有 value 都用字符串）：
{"origin_text": {"value": "...", "evidence": ""}, "residence_text": {"value": "...", "evidence": ""}, "nonlocal_origin_model": {"value": "...", "evidence": "..."}, "residence_local_model": {"value": "...", "evidence": "..."}, "status_at_judgment": {"value": "...", "evidence": "..."}, "ever_bail": {"value": "...", "evidence": "..."}, "plea_formal": {"value": "...", "evidence": "..."}, "jiejie": {"value": "...", "evidence": "..."}, "procedure": {"value": "...", "evidence": "..."}, "counsel_any": {"value": "...", "evidence": "..."}, "penalty_type": {"value": "...", "evidence": ""}, "term_months": {"value": "...", "evidence": ""}, "probation": {"value": "...", "evidence": ""}}
