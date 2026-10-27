// The views besides Ask AI: Dashboard, Compare companies, Extracted data, Evidence explorer and Settings.
import { api, pdfUrl } from "./api.js";
import { metricPanel } from "./charts.js";
import { factRef, openPage, passageRef, reportTitle, showEvidence } from "./evidence.js";
import { allCompaniesSelected, state } from "./state.js";
import {
  GAP_TEXT, MARGIN_NOTE, change, formatChange, formatNumber, formatValue, h, icon, metricName, previousFy, stateBox,
  swatch, unitLabel,
} from "./util.js";

const $ = (id) => document.getElementById(id);
const retryButton = (fn) => h("button", { type: "button", class: "btn btn-ghost btn-sm", onclick: () => fn() }, "Try again");
const count = new Intl.NumberFormat("en-IN");

// Each async render checks it is still the latest one, so quick filter changes never show stale results.
const runs = { dashboard: 0, data: 0, settings: 0, search: 0 };

// ---- Dashboard -----------------------------------------------------------------------------------

const KPIS = ["revenue", "net_profit", "operating_margin", "headcount", "attrition"];

export async function renderDashboard() {
  const run = ++runs.dashboard;
  const box = $("dashboard-body");
  $("dashboard-sub").textContent = state.year === "all"
    ? "The latest value in the extracted data for each figure. Select a value to see the page it came from."
    : `Figures for ${state.year}. Select a value to see the page it came from.`;
  if (!box.firstChild) box.replaceChildren(stateBox("loading", "Loading figures…"));
  let byMetric;
  try {
    const results = await Promise.all(KPIS.map((m) => api.metrics(m)));
    byMetric = Object.fromEntries(results.map((r) => [r.metric, r]));
  } catch (err) {
    if (run === runs.dashboard) box.replaceChildren(stateBox("error", "Could not load the figures", err.message, retryButton(renderDashboard)));
    return;
  }
  if (run !== runs.dashboard) return;
  box.replaceChildren(
    h("div", { class: "kpi-grid" }, state.companies.map((c) => companyCard(c, byMetric))),
    h("p", { class: "note" }, icon("info"), `${MARGIN_NOTE} Consolidated figures are used where a report gives both.`));
}

function companyCard(company, byMetric) {
  const report = state.overview?.reports?.find((r) => r.company === company);
  return h("article", { class: "card kpi-card", "aria-labelledby": `kpi-${company}` },
    h("header", { class: "kpi-head" }, swatch(company),
      h("div", { class: "kpi-head-text" }, h("h2", { id: `kpi-${company}` }, company),
        report ? h("p", { class: "kpi-report", title: report.title }, reportTitle({ company, report: report.title })) : "")),
    h("div", { class: "kpi-list" }, KPIS.map((m) => kpiTile(company, m, byMetric[m]))));
}

function kpiTile(company, metric, data) {
  const facts = data.facts.filter((f) => f.company === company).sort((a, b) => a.fy.localeCompare(b.fy));
  const fact = state.year === "all" ? facts.at(-1) : facts.find((f) => f.fy === state.year);
  const label = metric === "operating_margin" ? (company === "HCLTech" ? "EBIT margin" : "Operating margin") : metricName(metric);
  if (!fact) {
    return h("div", { class: "kpi is-gap", title: GAP_TEXT },
      h("span", { class: "kpi-label" }, label),
      h("span", { class: "kpi-value" }, h("span", { "aria-hidden": "true" }, "—"), h("span", { class: "visually-hidden" }, GAP_TEXT)),
      h("span", { class: "kpi-meta" }, state.year === "all" ? "Not in the extracted data" : `${state.year} · not in the extracted data`));
  }
  const prev = facts.find((f) => f.fy === previousFy(fact.fy));
  const delta = prev ? change(prev.value, fact.value, data.unit) : null;
  const [main, suffix] = splitValue(fact.value, data.unit);
  // The change is computed from two facts, so both are shown as evidence.
  const refs = [factRef(fact, metric), ...(prev ? [factRef(prev, metric)] : [])];
  return h("button", {
    type: "button", class: "kpi",
    "aria-label": `${company} ${label} ${fact.fy}: ${formatValue(fact.value, data.unit)}, page ${fact.page_no}. Show source`,
    onclick: () => showEvidence(refs, { view: "dashboard", focus: refs[0].key, open: true }),
  },
  h("span", { class: "kpi-label" }, label),
  h("span", { class: "kpi-value" }, main, suffix ? h("span", { class: "kpi-unit" }, ` ${suffix}`) : ""),
  h("span", { class: "kpi-meta" }, h("span", null, `${fact.fy} · p. ${fact.page_no}`),
    delta != null ? h("span", { class: "kpi-delta" }, `${delta > 0 ? "▲" : delta < 0 ? "▼" : "■"} ${formatChange(delta, data.unit)} vs ${prev.fy.slice(2)}`) : ""));
}

function splitValue(value, unit) {
  const n = formatNumber(value, unit);
  if (unit === "INR crore") return [`₹${n}`, "crore"];
  if (unit === "USD million") return [`US$${n}`, "million"];
  if (unit === "count") return [n, "employees"];
  return [n, ""];
}

// ---- Compare companies ---------------------------------------------------------------------------

let compareMetric = "revenue";
let comparePanel = null;

export function renderCompare() {
  comparePanel?.destroy();
  $("compare-sub").textContent = state.year === "all"
    ? "One metric across the selected companies, over every year in the extracted data. Computed in the browser from the facts; no model is involved."
    : `One metric across the selected companies in ${state.year}. Choose “All available years” in the top bar to see the trend.`;
  comparePanel = metricPanel({
    metric: compareMetric, companies: state.companies, year: state.year, layout: "stacked",
    onMetricChange: (m) => { compareMetric = m; },
    onPick: (facts, m) => {
      const refs = facts.map((f) => factRef(f, m));
      showEvidence(refs, { view: "compare", focus: refs[0].key, open: true });
    },
  });
  $("compare-body").replaceChildren(comparePanel.root);
}

// ---- Extracted data ------------------------------------------------------------------------------

const STATUS = {
  corroborated: { label: "Corroborated", icon: "check" },
  verified: { label: "Verified", icon: "check" },
  single_source: { label: "Single source", icon: "info" },
  conflict: { label: "Conflict", icon: "alert" },
  not_in_text: { label: "Not in text layer", icon: "alert" },
};

// The report's own label and the basis sit under the metric name, and the unit and page under the value, to keep the table narrow.
const COLUMNS = [
  { key: "company", label: "Company" },
  { key: "metric", label: "Metric" },
  { key: "fy", label: "Year" },
  { key: "value", label: "Value", num: true },
  { key: "status", label: "Status" },
  { key: "is_best", label: "Used in charts" },
];

const table = { sort: "company", dir: 1, metric: "", status: "", text: "", bestOnly: false };

export async function renderData() {
  const run = ++runs.data;
  const box = $("data-body");
  if (!box.firstChild) box.replaceChildren(stateBox("loading", "Loading extracted facts…"));
  let facts;
  try {
    facts = (await api.facts()).facts;
  } catch (err) {
    if (run === runs.data) box.replaceChildren(stateBox("error", "Could not load the extracted data", err.message, retryButton(renderData)));
    return;
  }
  if (run !== runs.data) return;
  box.replaceChildren(dataToolbar(facts), h("div", { id: "data-table-box" }));
  drawDataTable(facts);
}

function dataToolbar(facts) {
  const redraw = () => drawDataTable(facts);
  const statuses = [...new Set(facts.map((f) => f.status))];
  return h("div", { class: "toolbar" },
    h("label", { class: "field" }, h("span", null, "Metric"),
      h("select", { onchange: (e) => { table.metric = e.target.value; redraw(); } },
        h("option", { value: "" }, "All metrics"),
        state.metrics.map((m) => h("option", { value: m.metric, selected: table.metric === m.metric }, metricName(m.metric))))),
    h("label", { class: "field" }, h("span", null, "Status"),
      h("select", { onchange: (e) => { table.status = e.target.value; redraw(); } },
        h("option", { value: "" }, "All statuses"),
        statuses.map((s) => h("option", { value: s, selected: table.status === s }, STATUS[s]?.label || s)))),
    h("label", { class: "field grow" }, h("span", null, "Find"),
      h("input", { type: "search", placeholder: "Label or quote, e.g. Total employees", value: table.text,
        oninput: (e) => { table.text = e.target.value; redraw(); } })),
    h("label", { class: "check" },
      h("input", { type: "checkbox", checked: table.bestOnly, onchange: (e) => { table.bestOnly = e.target.checked; redraw(); } }),
      "Only values used in charts"));
}

function sortValue(f, key) {
  if (key === "is_best") return f.is_best ? 0 : 1;
  return f[key] ?? "";
}

function drawDataTable(facts) {
  const box = $("data-table-box");
  if (!box) return;
  const needle = table.text.trim().toLowerCase();
  const rows = facts
    .filter((f) => state.companies.includes(f.company)
      && (state.year === "all" || f.fy === state.year)
      && (!table.metric || f.metric === table.metric)
      && (!table.status || f.status === table.status)
      && (!table.bestOnly || f.is_best)
      && (!needle || `${f.label_in_report} ${f.quote}`.toLowerCase().includes(needle)))
    .sort((a, b) => {
      const x = sortValue(a, table.sort);
      const y = sortValue(b, table.sort);
      const primary = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return primary * table.dir || a.company.localeCompare(b.company) || a.metric.localeCompare(b.metric) || a.fy.localeCompare(b.fy);
    });

  const head = h("tr", null, COLUMNS.map((col) => {
    const sorted = table.sort === col.key;
    return h("th", { scope: "col", class: col.num ? "num" : null, "aria-sort": sorted ? (table.dir > 0 ? "ascending" : "descending") : null },
      h("button", { type: "button", class: "sort-btn", onclick: () => {
        table.dir = sorted ? -table.dir : 1;
        table.sort = col.key;
        drawDataTable(facts);
        $("data-table-box").querySelector(`th[aria-sort] .sort-btn`)?.focus();
      } }, col.label, icon("sort", `icon sort-icon${sorted ? " on" : ""}`)));
  }));

  const body = rows.map((f) => {
    const open = () => {
      const ref = factRef(f);
      showEvidence([ref], { view: "data", focus: ref.key, open: true });
    };
    const status = STATUS[f.status] || { label: f.status, icon: "info" };
    const basis = f.basis === "unknown" ? "" : ` · ${f.basis}`;
    return h("tr", { class: "clickable", onclick: (e) => { if (!e.target.closest("button")) open(); } },
      h("td", null, h("span", { class: "company-cell" }, swatch(f.company), f.company)),
      h("td", null,
        h("span", { class: "cell-main" }, metricName(f.metric)),
        h("span", { class: "cell-sub truncate", title: `Label in report: ${f.label_in_report}${basis}` }, `${f.label_in_report}${basis}`)),
      h("td", { class: "nowrap" }, f.fy),
      h("td", { class: "num" },
        h("button", { type: "button", class: "num-btn", "aria-label": `${f.company} ${metricName(f.metric)} ${f.fy}: ${formatValue(f.value, f.unit)}. Show source`, onclick: open },
          formatNumber(f.value, f.unit)),
        h("span", { class: "cell-sub" }, f.unit === "%" ? `p. ${f.page_no}` : `${unitLabel(f.unit)} · p. ${f.page_no}`)),
      h("td", null, h("span", { class: `status-chip s-${f.status}` }, icon(status.icon), status.label)),
      h("td", null, f.is_best ? h("span", { class: "best-chip", title: "Used in charts and answers" }, icon("star"), "Used") : h("span", { class: "visually-hidden" }, "Not used")));
  });

  const conflicts = rows.filter((f) => f.status === "conflict").length;
  box.replaceChildren(
    h("p", { class: "table-summary" },
      `${rows.length} of ${facts.length} facts · ${rows.filter((f) => f.is_best).length} used in charts · ${conflicts} conflict${conflicts === 1 ? "" : "s"}`),
    rows.length
      ? h("div", { class: "table-wrap card" }, h("table", { class: "data-table facts-table" }, h("thead", null, head), h("tbody", null, body)))
      : stateBox("empty", "No facts match these filters", "Clear a filter, or choose more companies or All available years in the top bar."));
}

// ---- Evidence explorer ---------------------------------------------------------------------------

const EXAMPLES = ["generative AI", "voluntary attrition", "dividend", "Americas", "carbon emissions"];
const search = { query: "", companies: "", results: null, error: null };

export function initExplorer() {
  $("search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch($("search-input").value);
  });
}

async function runSearch(raw) {
  const query = raw.trim();
  if (!query) return;
  $("search-input").value = query;
  Object.assign(search, { query, companies: state.companies.join(), results: null, error: null });
  const run = ++runs.search;
  $("explorer-body").replaceChildren(stateBox("loading", `Searching for “${query}”…`));
  try {
    const lists = allCompaniesSelected()
      ? [await api.search(query)]
      : await Promise.all(state.companies.map((c) => api.search(query, c)));
    search.results = lists.flatMap((l) => l.results).sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  } catch (err) {
    search.error = err.message;
  }
  if (run === runs.search) renderExplorer();
}

export function renderExplorer() {
  const box = $("explorer-body");
  if (search.query && search.companies !== state.companies.join()) return runSearch(search.query);   // filter changed
  if (search.error) return box.replaceChildren(stateBox("error", "Search failed", search.error, retryButton(() => runSearch(search.query))));
  if (!search.results) {
    return box.replaceChildren(h("div", { class: "card examples" },
      h("p", null, "Try one of these searches:"),
      h("div", { class: "chip-row" }, EXAMPLES.map((q) => h("button", { type: "button", class: "chip", onclick: () => runSearch(q) }, q)))));
  }
  if (!search.results.length) {
    return box.replaceChildren(stateBox("empty", `No passages match “${search.query}”`, "Try fewer or different words."));
  }
  box.replaceChildren(
    h("p", { class: "table-summary" }, `${search.results.length} passages for “${search.query}”, best matches first`),
    h("ol", { class: "results" }, search.results.map(resultItem)));
}

function resultItem(r) {
  const ref = passageRef(r);
  const open = async (event) => {
    const from = event.currentTarget;
    showEvidence([ref], { view: "explorer", focus: ref.key, open: true });
    const ev = await api.evidence(ref).catch(() => null);   // has the passage's box on the page
    openPage(ev ? { ...ref, bbox: ev.bbox } : ref, from);
  };
  return h("li", null, h("button", { type: "button", class: "result card", onclick: open },
    h("span", { class: "result-head" },
      swatch(r.company),
      h("span", { class: "result-company" }, r.company),
      h("span", { class: "result-report" }, r.report),
      h("span", { class: "kind-chip" }, r.kind === "table" ? "Table" : "Text"),
      h("span", { class: "badge" }, `Page ${r.page}`)),
    r.section ? h("span", { class: "result-section" }, r.section) : "",
    h("span", { class: "result-snippet" }, highlight(r.snippet))));
}

/** The API marks matched words as [word]; show them as <mark>, as text nodes only. */
function highlight(snippet) {
  return snippet.split(/\[([^\]]+)\]/).map((part, i) => (i % 2 ? h("mark", null, part) : document.createTextNode(part)));
}

// ---- Settings ------------------------------------------------------------------------------------

const BACKEND_NAMES = {
  vllm: "vLLM (self-hosted)",
  transformers: "Transformers, self-hosted",
  extractive: "Extractive (no model)",
  facts: "Extracted data",
};

export async function renderSettings() {
  const run = ++runs.settings;
  const box = $("settings-body");
  if (!box.firstChild) box.replaceChildren(stateBox("loading", "Loading…"));
  let overview;
  let health = null;
  try {
    [overview, health] = await Promise.all([api.overview(), api.health().catch(() => null)]);
  } catch (err) {
    if (run === runs.settings) box.replaceChildren(stateBox("error", "Could not load the settings", err.message, retryButton(renderSettings)));
    return;
  }
  if (run !== runs.settings) return;
  const c = overview.counts || {};
  box.replaceChildren(h("div", { class: "settings-grid" },
    settingsCard("Answering model", "Metric lookups and comparisons are answered straight from the extracted data; other questions go to the model, which answers only from retrieved report passages.",
      [["Backend", BACKEND_NAMES[overview.llm_backend] || overview.llm_backend],
        ["Model", overview.model || "—"],
        ["Server status", health ? (health.status === "ok" ? "Running" : health.status) : "Unknown"]]),
    settingsCard("Extracted data", "Built once from the three PDFs, without OCR.",
      [["Facts", count.format(c.facts ?? 0)], ["Passages", count.format(c.passages ?? 0)],
        ["Tables", count.format(c.tables ?? 0)], ["Pages", count.format(c.pages ?? 0)]]),
    h("section", { class: "card settings-card wide" },
      h("h2", null, "Reports"),
      h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
        h("thead", null, h("tr", null, ["Company", "Report", "Year", "Pages", ""].map((t, i) => h("th", { scope: "col", class: i === 3 ? "num" : null }, t)))),
        h("tbody", null, overview.reports.map((r) => h("tr", null,
          h("td", null, h("span", { class: "company-cell" }, swatch(r.company), r.company)),
          h("td", null, r.title),
          h("td", { class: "nowrap" }, r.fy),
          h("td", { class: "num" }, count.format(r.pages)),
          h("td", null, h("a", { class: "text-link", href: pdfUrl(r.pdf, 1), target: "_blank", rel: "noopener" }, "Open PDF", icon("external"))))))))),
    h("section", { class: "card settings-card wide" },
      h("h2", null, "Metrics"),
      h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
        h("thead", null, h("tr", null, ["Metric", "As labelled in the reports", "Unit"].map((t) => h("th", { scope: "col" }, t)))),
        h("tbody", null, overview.metrics.map((m) => h("tr", null,
          h("td", null, metricName(m.metric)), h("td", null, m.label), h("td", null, unitLabel(m.unit))))))))));
}

function settingsCard(title, text, rows) {
  return h("section", { class: "card settings-card" },
    h("h2", null, title),
    h("p", { class: "muted" }, text),
    h("dl", { class: "facts-list" }, rows.map(([k, v]) => h("div", null, h("dt", null, k), h("dd", null, v)))));
}
