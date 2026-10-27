"""Generates the SYNTHETIC fixture (fictional companies, invented numbers). Not real annual-report data.
Run: python data/make_synthetic_fixture.py   -> data/synthetic_corpus.jsonl, data/synthetic_queries.jsonl
Labels are derived from chunk ids by construction (provenance="synthetic"); they say nothing about real-report quality."""
import json, random
from pathlib import Path

FIXTURE_VERSION = "v2"

HERE = Path(__file__).parent
COS = {"Acme": "Acme Infotech", "Borealis": "Borealis Software", "Cygnus": "Cygnus Digital"}
FYS = ["FY2025-26", "FY2024-25"]
PRIOR = {"FY2025-26": "FY2024-25", "FY2024-25": "FY2023-24"}
STRATEGY = {
    "Acme": "generative AI platform engineering and automation of delivery pipelines",
    "Borealis": "cybersecurity services and large-scale cloud migration programmes",
    "Cygnus": "data analytics, data engineering and industry-specific digital products",
}
RISKS = {
    "Acme": "currency volatility, dependence on a few large clients and delivery cost inflation",
    "Borealis": "data-breach exposure, regulatory change in key markets and vendor concentration",
    "Cygnus": "pricing pressure, slower discretionary spending by clients and talent shortages",
}

def sid(co, fy): return f"{co.lower()}-{fy.replace('FY','fy').replace('-','')[:6]}"
def fmt(n): return f"{int(n):,}"

chunks, doc_page = [], {}

def add(cid, text, md): chunks.append({"chunk_id": cid, "text": text, "metadata": md})

for co in COS:
    for fy in FYS:
        rng = random.Random(f"{co}-{fy}")
        doc = f"{co.lower()}_{fy}_ar"
        page = [10]
        def nextpage():
            page[0] += rng.randint(1, 4); return page[0]
        prior = PRIOR[fy]
        base = dict(company=co, fiscal_year=fy, doc_id=doc)
        facts = {}
        for scope in ("consolidated", "standalone"):
            k = 1.0 if scope == "consolidated" else 0.72
            rev = rng.randint(20000, 90000) * k; oth = rng.randint(500, 3000) * k
            pbt = rev * rng.uniform(0.19, 0.27); pat = pbt * 0.75
            prev = rev / rng.uniform(1.05, 1.14)
            pg = nextpage()
            tid = f"tbl-{sid(co,fy)}-{scope[:4]}-pl"
            rows = [("Revenue from operations", rev, prev), ("Other income", oth, oth * 0.9),
                    ("Total income", rev + oth, prev + oth * 0.9), ("Profit before tax", pbt, pbt / 1.1),
                    ("Profit for the year", pat, pat / 1.1)]
            slug = ["rev", "oth", "tot", "pbt", "pat"]
            for i, ((lab, cur, pr), sl) in enumerate(zip(rows, slug)):
                add(f"{sid(co,fy)}-{scope[:4]}-{sl}", f"{lab} | {fmt(cur)} | {fmt(pr)}",
                    dict(base, pdf_page_index=pg, printed_page=pg - 6, content_type="table", scope=scope, table_id=tid,
                         table_title=f"{scope.capitalize()} Statement of Profit and Loss (extract)",
                         headers=f"Particulars|{fy}|{prior}", row_label=lab, row_index=i, unit="INR crore",
                         periods=f"{fy}|{prior}"))
            facts[scope] = pg
        pg = nextpage(); tid = f"tbl-{sid(co,fy)}-kpi"
        hc = rng.randint(90000, 600000); om = rng.uniform(18, 26); attr = rng.uniform(9, 20); ob = rng.randint(6000, 14000)
        kp = [("Operating margin (%)", f"{om:.1f}", f"{om-0.4:.1f}", "ebit-margin", "%"),
              ("Employee headcount (number)", fmt(hc), fmt(hc * 0.97), "headcount", "number"),
              ("Attrition rate (%)", f"{attr:.1f}", f"{attr+2:.1f}", "attrition", "%"),
              ("Order book (USD million)", fmt(ob), fmt(ob * 0.92), "orderbook", "USD million")]
        for i, (lab, c, p, sl, un) in enumerate(kp):
            add(f"{sid(co,fy)}-kpi-{sl}", f"{lab} | {c} | {p}",
                dict(base, pdf_page_index=pg, printed_page=pg - 6, content_type="table", scope="consolidated", table_id=tid,
                     table_title="Key Performance Indicators", headers=f"Indicator|{fy}|{prior}", row_label=lab,
                     row_index=i, unit=un, periods=f"{fy}|{prior}"))
        facts["kpi"] = pg
        N = COS[co]
        def nar(slug, text):
            p = nextpage()
            add(f"{sid(co,fy)}-nar-{slug}", text, dict(base, pdf_page_index=p, printed_page=p - 6, content_type="narrative"))
            facts[slug] = p
        nar("people", f"People: {N} ended {fy} with a workforce of {fmt(hc)} employees. Voluntary attrition was {attr:.1f} percent, "
                      f"down from {attr+2:.1f} percent in the previous year, as retention programmes took effect.")
        nar("strategy", f"Strategy: In {fy} {N} concentrated its investments on {STRATEGY[co]}. Management expects these areas to drive "
                        f"the next phase of growth and has realigned its leadership structure around them.")
        nar("risk", f"Risk management: The principal risks identified by {N} are {RISKS[co]}. The audit committee reviews mitigation "
                    f"plans every quarter.")
        nar("dividend", f"Dividend: The Board of {N} recommended a final dividend of Rs {rng.randint(10, 40)} per share for {fy}, "
                        f"subject to shareholder approval at the annual general meeting.")
        nar("deals", f"Deal momentum: Total contract value of large deal wins in {fy} reached USD {fmt(ob*1.1)} million, with several "
                     f"multi-year renewals in banking and retail.")
        nar("governance", f"Governance: The board of {N} held six meetings during {fy}. All independent directors met the criteria "
                          f"of independence prescribed under the listing regulations.")
        nar("sustainability", f"Sustainability: {N} continued its programmes on energy efficiency, responsible sourcing and community "
                              f"education. Detailed environmental disclosures follow the voluntary reporting framework.")
        nar("outlook", f"Outlook: Demand for technology services remains healthy but clients continue to scrutinise discretionary "
                       f"budgets. {N} will keep investing in skills and delivery tooling.")
        doc_page[(co, fy)] = facts

# --- v2: appended policy chunks (existing ids/order untouched). Some deliberately omit the company name and/or the
# literal fiscal-year string so retrieval must rely on metadata, not on text overlap. Numbers are invented.
POLICY = {
    ("Acme", "FY2025-26"): (
        "Dividend policy: The Board of Acme Infotech follows a policy of distributing about 35 percent of profit after tax as dividend, "
        "while keeping adequate reserves for investment.",
        "Risk policy: Foreign-exchange exposure is hedged on a rolling basis, covering between 40 and 60 percent of forecast dollar "
        "receivables for the next twelve months."),
    ("Acme", "FY2024-25"): (
        "Dividend policy: The Board of Acme Infotech follows a policy of distributing about 30 percent of profit for the year ended "
        "March 2025 as dividend, retaining the rest to fund expansion.",
        "Risk policy: Acme Infotech hedges between 30 and 50 percent of forecast dollar receivables for the next twelve months."),
    ("Borealis", "FY2025-26"): (
        "Dividend policy: The Board follows a policy of distributing about 40 percent of profit after tax to shareholders once the "
        "capital needs for acquisitions are met.",
        "Risk policy: Cyber-incident exposure is covered by an insurance programme of USD 150 million and an annual third-party "
        "penetration test of every client-facing platform."),
    ("Borealis", "FY2024-25"): (
        "Dividend policy: The Board follows a policy of distributing about 38 percent of profit after tax to shareholders, after "
        "setting aside funds for acquisitions.",
        "Risk policy: Borealis Software carries cyber insurance of USD 120 million and runs penetration tests on client-facing "
        "platforms every year in FY2024-25."),
    ("Cygnus", "FY2025-26"): (
        "Dividend policy: Cygnus Digital intends to pay out roughly 25 percent of profit after tax each year, reviewed annually by "
        "the Board.",
        "Risk policy: Pricing and talent risks are managed through caps on fixed-price contracts and a funded retention budget for "
        "critical data-engineering skills."),
    ("Cygnus", "FY2024-25"): (
        "Dividend policy: Cygnus Digital targets a payout of roughly 22 percent of profit after tax in FY2024-25, reviewed annually.",
        "Risk policy: Cygnus Digital limits fixed-price work to one fifth of bookings and funds a retention budget for scarce skills."),
}
for co in COS:
    for fy in FYS:
        base = dict(company=co, fiscal_year=fy, doc_id=f"{co.lower()}_{fy}_ar")
        for off, (slug, text) in enumerate(zip(("divpolicy", "riskpolicy"), POLICY[(co, fy)]), start=1):
            p = doc_page[(co, fy)]["outlook"] + off
            add(f"{sid(co,fy)}-nar-{slug}", text, dict(base, pdf_page_index=p, printed_page=p - 6, content_type="narrative"))
            doc_page[(co, fy)][slug] = p

Q = []
TPL = [None]
def T(t): TPL[0] = t   # legacy loop-generated queries: one template id per f-string template
def q(cat, question, qrels, companies, fy, scope=None, tables=(), rows=(), pages=()):
    Q.append({"qid": f"q{len(Q)+1:03d}", "question": question, "category": cat, "template": TPL[0], "qrels": dict(qrels),
              "expected": {k: v for k, v in dict(companies=list(companies), fiscal_year=fy, scope=scope,
                                                  table_ids=list(tables), row_chunk_ids=list(rows),
                                                  pages=[list(p) for p in pages]).items() if v},
              "provenance": "synthetic"})
def cid(co, fy, part): return f"{sid(co,fy)}-{part}"
def doc(co, fy): return f"{co.lower()}_{fy}_ar"
def pl_tid(co, fy, scope): return f"tbl-{sid(co,fy)}-{scope[:4]}-pl"
def pg(co, fy, key): return (doc(co, fy), doc_page[(co, fy)][key])
F = "FY2025-26"; G = "FY2024-25"
cons, sta = "consolidated", "standalone"

for co in COS:
    T("T01")
    q("numeric_lookup", f"What was {co}'s consolidated revenue from operations in {F}?", {cid(co,F,"cons-rev"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-rev")], [pg(co,F,cons)])
    T("T02")
    q("numeric_lookup", f"What was {co}'s profit for the year on a consolidated basis in {F}?", {cid(co,F,"cons-pat"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-pat")], [pg(co,F,cons)])
    T("T03")
    q("numeric_lookup", f"How many employees did {co} have at the end of {F}?",
      {cid(co,F,"kpi-headcount"): 3, cid(co,F,"nar-people"): 2}, [co], F, None,
      [f"tbl-{sid(co,F)}-kpi"], [cid(co,F,"kpi-headcount")], [pg(co,F,"kpi"), pg(co,F,"people")])
for co in ("Acme", "Borealis"):
    T("T04")
    q("terminology", f"{co} total income consolidated {F}", {cid(co,F,"cons-tot"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-tot")], [pg(co,F,cons)])
    T("T05")
    q("terminology", f"{co} other income consolidated {F}", {cid(co,F,"cons-oth"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-oth")], [pg(co,F,cons)])
    T("T06")
    q("terminology", f"{co} revenue from operations {F} consolidated statement", {cid(co,F,"cons-rev"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-rev")], [pg(co,F,cons)])
for co in COS:
    T("T07")
    q("paraphrase", f"How much net sales did {co} book in {F} on a consolidated basis?", {cid(co,F,"cons-rev"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-rev")], [pg(co,F,cons)])
    T("T08")
    q("paraphrase", f"What was the bottom line for {co} in {F}, consolidated?", {cid(co,F,"cons-pat"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-pat")], [pg(co,F,cons)])
    T("T09")
    q("paraphrase", f"What is the size of {co}'s workforce in {F}?",
      {cid(co,F,"kpi-headcount"): 3, cid(co,F,"nar-people"): 2}, [co], F, None,
      [f"tbl-{sid(co,F)}-kpi"], [cid(co,F,"kpi-headcount")], [pg(co,F,"kpi"), pg(co,F,"people")])
    T("T10")
    q("paraphrase", f"How much did {co} earn from selling its services in {F}, consolidated?", {cid(co,F,"cons-rev"): 3}, [co], F, cons,
      [pl_tid(co,F,cons)], [cid(co,F,"cons-rev")], [pg(co,F,cons)])
for co in COS:
    T("T11")
    q("historical", f"What was {co}'s consolidated revenue from operations in {G}?",
      {cid(co,G,"cons-rev"): 3, cid(co,F,"cons-rev"): 2}, [co], G, cons,
      [pl_tid(co,G,cons), pl_tid(co,F,cons)], [cid(co,G,"cons-rev")], [pg(co,G,cons), pg(co,F,cons)])
    T("T12")
    q("historical", f"How did {co}'s attrition rate change compared with the previous year in {F}?",
      {cid(co,F,"nar-people"): 3, cid(co,F,"kpi-attrition"): 2}, [co], F, None,
      [f"tbl-{sid(co,F)}-kpi"], [cid(co,F,"kpi-attrition")], [pg(co,F,"people"), pg(co,F,"kpi")])
for a, b in (("Acme", "Borealis"), ("Borealis", "Cygnus"), ("Acme", "Cygnus")):
    T("T13")
    q("multi_company", f"Compare consolidated revenue from operations of {a} and {b} in {F}",
      {cid(a,F,"cons-rev"): 3, cid(b,F,"cons-rev"): 3}, [a, b], F, cons,
      [pl_tid(a,F,cons), pl_tid(b,F,cons)], [cid(a,F,"cons-rev"), cid(b,F,"cons-rev")], [pg(a,F,cons), pg(b,F,cons)])
T("T14")
q("multi_company", f"Which company had the highest consolidated profit for the year in {F}: Acme, Borealis or Cygnus?",
  {cid(c,F,"cons-pat"): 3 for c in COS}, list(COS), F, cons, [pl_tid(c,F,cons) for c in COS], [cid(c,F,"cons-pat") for c in COS],
  [pg(c,F,cons) for c in COS])
T("T15")
q("multi_company", f"Compare the headcount of Acme and Cygnus in {F}",
  {cid("Acme",F,"kpi-headcount"): 3, cid("Cygnus",F,"kpi-headcount"): 3, cid("Acme",F,"nar-people"): 2, cid("Cygnus",F,"nar-people"): 2},
  ["Acme", "Cygnus"], F, None, [f"tbl-{sid('Acme',F)}-kpi", f"tbl-{sid('Cygnus',F)}-kpi"],
  [cid("Acme",F,"kpi-headcount"), cid("Cygnus",F,"kpi-headcount")], [pg("Acme",F,"kpi"), pg("Cygnus",F,"kpi")])
T("T16")
q("multi_company", f"Compare attrition between Borealis and Cygnus in {F}",
  {cid("Borealis",F,"kpi-attrition"): 3, cid("Cygnus",F,"kpi-attrition"): 3, cid("Borealis",F,"nar-people"): 2,
   cid("Cygnus",F,"nar-people"): 2}, ["Borealis", "Cygnus"], F, None,
  [f"tbl-{sid('Borealis',F)}-kpi", f"tbl-{sid('Cygnus',F)}-kpi"],
  [cid("Borealis",F,"kpi-attrition"), cid("Cygnus",F,"kpi-attrition")], [pg("Borealis",F,"kpi"), pg("Cygnus",F,"kpi")])
for co in ("Acme", "Cygnus"):
    T("T17")
    q("scope", f"What was {co}'s standalone revenue from operations in {F}?", {cid(co,F,"stan-rev"): 3}, [co], F, sta,
      [pl_tid(co,F,sta)], [cid(co,F,"stan-rev")], [pg(co,F,sta)])
for co in ("Borealis", "Cygnus"):
    T("T18")
    q("scope", f"{co} standalone profit for the year {F}", {cid(co,F,"stan-pat"): 3}, [co], F, sta,
      [pl_tid(co,F,sta)], [cid(co,F,"stan-pat")], [pg(co,F,sta)])
for co in COS:
    T("T19")
    q("qualitative", f"What is {co}'s strategic focus for {F}?", {cid(co,F,"nar-strategy"): 3}, [co], F, None, [], [], [pg(co,F,"strategy")])
    T("T20")
    q("qualitative", f"What risks does {co} highlight in {F}?", {cid(co,F,"nar-risk"): 3}, [co], F, None, [], [], [pg(co,F,"risk")])
n = 20
for text in (f"What were Acme's Scope 1 carbon emissions in {F}?", "What is Borealis's share price target?",
             "How many patents did Cygnus file?", f"What was Acme's revenue from the Brazil market in {F}?"):
    n += 1
    Q.append({"qid": f"q{len(Q)+1:03d}", "question": text, "category": "unanswerable", "template": f"T{n:02d}", "qrels": {},
              "expected": {}, "provenance": "synthetic"})

# ---------------------------------------------------------------------------------------------------------------------
# v2: hand-written queries (each its own template id H01...). Conventions:
#  * no year cue and no "latest" cue -> FY2025-26 chunks only (FY2024-25 absent from qrels); same for explicit latest cues
#  * scope omitted on P&L items -> consolidated row grade 3, standalone row grade 2 (KPI rows are consolidated-only)
#  * two-year change questions -> both years' rows grade 3, expected.fiscal_year omitted
#  * a year-ended-earlier question keeps the existing legacy pattern (asked year 3, following-year row 2)
H = [0]
ALL = list(COS)
def pk(part):
    if part.startswith("cons-"): return cons
    if part.startswith("stan-"): return sta
    if part.startswith("kpi-"): return "kpi"
    return part.split("-", 1)[1]
def hq(cat, question, parts, fy=None, scope=None):
    """parts: list of (company, fiscal_year, part, grade); expected block is derived from the parts."""
    H[0] += 1; T(f"H{H[0]:02d}")
    qr, cos, tabs, rws, pgs = {}, [], [], [], []
    for co, y, part, g in parts:
        qr[cid(co, y, part)] = g
        if co not in cos: cos.append(co)
        if part.startswith(("cons-", "stan-")):
            t = pl_tid(co, y, cons if part.startswith("cons-") else sta)
        elif part.startswith("kpi-"):
            t = f"tbl-{sid(co,y)}-kpi"
        else:
            t = None
        if t and t not in tabs: tabs.append(t)
        if t and g == 3: rws.append(cid(co, y, part))
        p = pg(co, y, pk(part))
        if p not in pgs: pgs.append(p)
    q(cat, question, qr, cos, fy, scope, tabs, rws, pgs)
def hu(question):
    H[0] += 1; T(f"H{H[0]:02d}")
    Q.append({"qid": f"q{len(Q)+1:03d}", "question": question, "category": "unanswerable", "template": TPL[0], "qrels": {},
              "expected": {}, "provenance": "synthetic"})
def each(part, g=3, y=F): return [(c, y, part, g) for c in ALL]

# 1. comparisons without named companies
hq("multi_company", "Which company had the highest operating margin?", each("kpi-ebit-margin"))
hq("multi_company", "Who reported the biggest workforce of the lot in FY2025-26?",
   each("kpi-headcount") + each("nar-people", 2), F)
hq("multi_company", "Order book: whose is the largest?", each("kpi-orderbook") + each("nar-deals", 1))
hq("multi_company", "Among these businesses, where did staff leave at the steepest rate in FY2025-26?",
   each("kpi-attrition") + each("nar-people", 2), F)
hq("multi_company", "Which one posted the lowest profit for the year, consolidated, in FY2025-26?", each("cons-pat"), F, cons)
hq("multi_company", "Which firm booked the largest revenue from operations?", each("cons-rev") + each("stan-rev", 2))
hq("multi_company", "Rank everyone by profit after tax.", each("cons-pat") + each("stan-pat", 2))
# 2. implicit latest period
hq("numeric_lookup", "What is Acme's latest reported order book?", [("Acme", F, "kpi-orderbook", 3), ("Acme", F, "nar-deals", 1)])
hq("numeric_lookup", "Give me Borealis headcount for the most recent year.",
   [("Borealis", F, "kpi-headcount", 3), ("Borealis", F, "nar-people", 2)])
hq("numeric_lookup", "Cygnus staff turnover in the current year",
   [("Cygnus", F, "kpi-attrition", 3), ("Cygnus", F, "nar-people", 2)])
hq("numeric_lookup", "In the most recent annual report, what operating margin did each company disclose?", each("kpi-ebit-margin"))
hq("multi_company", "Consolidated revenue from operations for every company, latest year", each("cons-rev"), None, cons)
hq("numeric_lookup", "Latest reported profit for Borealis",
   [("Borealis", F, "cons-pat", 3), ("Borealis", F, "stan-pat", 2)])
hq("qualitative", "Current-year dividend policy at each company", each("nar-divpolicy") + each("nar-dividend", 2))
# 3. narrative: risks and policies
hq("qualitative", "What is Borealis's dividend policy?", [("Borealis", F, "nar-divpolicy", 3), ("Borealis", F, "nar-dividend", 1)])
hq("qualitative", "How much of its earnings does Cygnus hand to shareholders under its payout policy in FY2024-25?",
   [("Cygnus", G, "nar-divpolicy", 3), ("Cygnus", G, "nar-dividend", 2)], G)
hq("qualitative", "Which company distributes the largest share of profit to its shareholders as a matter of policy?",
   each("nar-divpolicy"))
hq("qualitative", "How does Acme protect itself against currency swings?",
   [("Acme", F, "nar-riskpolicy", 3), ("Acme", F, "nar-risk", 2)])
hq("qualitative", "What safeguards does Borealis have against cyber incidents?",
   [("Borealis", F, "nar-riskpolicy", 3), ("Borealis", F, "nar-risk", 2)])
hq("qualitative", "Describe how Cygnus manages pricing and talent risk in FY2025-26",
   [("Cygnus", F, "nar-riskpolicy", 3), ("Cygnus", F, "nar-risk", 2)], F)
hq("qualitative", "Main risks the audit committee reviews every quarter?", each("nar-risk"))
hq("qualitative", "What final dividend per share did Acme's Board put forward for FY2024-25?",
   [("Acme", G, "nar-dividend", 3), ("Acme", G, "nar-divpolicy", 1)], G)
# 4. scope omitted
hq("scope", "What did Acme earn as profit for the year in FY2025-26?", [("Acme", F, "cons-pat", 3), ("Acme", F, "stan-pat", 2)], F)
hq("scope", "Revenue from operations for Cygnus, FY2025-26", [("Cygnus", F, "cons-rev", 3), ("Cygnus", F, "stan-rev", 2)], F)
hq("scope", "Total income that Borealis reported for the year ended March 2026",
   [("Borealis", F, "cons-tot", 3), ("Borealis", F, "stan-tot", 2)], F)
hq("scope", "How big was Cygnus's other income in fiscal 2026?", [("Cygnus", F, "cons-oth", 3), ("Cygnus", F, "stan-oth", 2)], F)
# 5. fiscal-year paraphrases and historical comparisons
hq("paraphrase", "Acme operating margin in FY26?", [("Acme", F, "kpi-ebit-margin", 3)], F)
hq("paraphrase", "Borealis employee strength, fiscal 2026",
   [("Borealis", F, "kpi-headcount", 3), ("Borealis", F, "nar-people", 2)], F)
hq("paraphrase", "Consolidated revenue from operations of Cygnus for FY25-26", [("Cygnus", F, "cons-rev", 3)], F, cons)
hq("paraphrase", "Profit for the year ended March 2026 for Acme on a consolidated basis", [("Acme", F, "cons-pat", 3)], F, cons)
hq("historical", "Cygnus consolidated revenue from operations back in FY 24-25",
   [("Cygnus", G, "cons-rev", 3), ("Cygnus", F, "cons-rev", 2)], G, cons)
hq("historical", "How did Borealis's consolidated profit for the year change from FY2024-25 to FY2025-26?",
   [("Borealis", G, "cons-pat", 3), ("Borealis", F, "cons-pat", 3)], None, cons)
hq("historical", "Acme's order book movement between FY24-25 and FY25-26",
   [("Acme", G, "kpi-orderbook", 3), ("Acme", F, "kpi-orderbook", 3)])
hq("historical", "Set Cygnus's headcount for the year ended March 2025 beside the year ended March 2026",
   [("Cygnus", G, "kpi-headcount", 3), ("Cygnus", F, "kpi-headcount", 3)])
# 7. evidence needed from all three companies
hq("multi_company", "Across Acme, Borealis and Cygnus, what was the total contract value of large deal wins in FY2025-26?",
   each("nar-deals") + each("kpi-orderbook", 2), F)
hq("multi_company", "Summarise where Acme, Borealis and Cygnus are concentrating investment in FY2025-26", each("nar-strategy"), F)
hq("multi_company", "List the final dividend per share each of the three firms recommended for FY2025-26", each("nar-dividend"), F)
hq("multi_company", "What total income did Acme, Borealis and Cygnus each report on a consolidated basis in FY2025-26?",
   each("cons-tot"), F, cons)
# 6. unanswerable (topics absent from the corpus)
hu("How many data centres does Borealis operate?")
hu("What net promoter score did Cygnus's customers give?")
hu("Who is the chief financial officer of Acme?")
hu("At what exercise price were employee stock options granted by Acme in FY2025-26?")
hu("Which of these companies has the greatest share of women in senior leadership?")

(HERE / "synthetic_corpus.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
(HERE / "synthetic_queries.jsonl").write_text("\n".join(json.dumps(x) for x in Q) + "\n")
print(f"fixture {FIXTURE_VERSION}: wrote {len(chunks)} chunks, {len(Q)} queries")
