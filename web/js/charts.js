// One metric across companies: a chart, a table with growth, and key insights.
// Everything is computed here from the facts /api/metrics returns; no value is ever made up or interpolated.
import { api } from "./api.js";
import { tabKeys } from "./evidence.js";
import { metricInfo, state } from "./state.js";
import {
  COMPANY_COLORS, FONT_STACK, GAP_TEXT, MARGIN_NOTE, POINT_STYLES, cagr, change, formatChange, formatNumber,
  formatValue, fyEnd, h, icon, metricName, previousFy, stateBox, swatch, unitLabel,
} from "./util.js";

// ---- Data helpers --------------------------------------------------------------------------------

/** company -> fy -> fact, for the given companies. */
export function gridOf(facts, companies) {
  const grid = Object.fromEntries(companies.map((c) => [c, {}]));
  for (const f of facts) if (grid[f.company]) grid[f.company][f.fy] = f;
  return grid;
}

export const yearsOf = (grid) => [...new Set(Object.values(grid).flatMap((byFy) => Object.keys(byFy)))].sort();

const listing = (items) => items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;

/** The not-available marker: a dash with the reason as a tooltip and for screen readers. */
export function gap() {
  return h("span", { class: "gap", title: GAP_TEXT }, h("span", { "aria-hidden": "true" }, "—"), h("span", { class: "visually-hidden" }, GAP_TEXT));
}

/** A number that opens its evidence when selected. */
function numberButton(text, facts, onPick, label) {
  return h("button", { type: "button", class: "num-btn", title: label, "aria-label": label, onclick: () => onPick(facts) }, text);
}

const pageList = (facts) => [...new Set(facts.map((f) => f.page_no))].map((p) => `p. ${p}`).join(", ");

// ---- Chart ---------------------------------------------------------------------------------------

// Fill the canvas white so a downloaded PNG is not transparent.
const whiteBackground = {
  id: "whiteBackground",
  beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.globalCompositeOperation = "destination-over";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  },
};

let chartSetup = false;
function setupChartJs() {
  if (chartSetup) return;
  chartSetup = true;
  const Chart = window.Chart;
  Chart.defaults.font.family = FONT_STACK;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#5D6679";
  Chart.defaults.animation = false;   // values appear as they are, never growing from zero
  Chart.register(whiteBackground);
}

const axisNumber = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 });

export function drawChart(canvas, { type, unit, companies, years, grid, onPick, legendBelow }) {
  setupChartJs();
  const datasets = companies.map((c) => ({
    label: c,
    data: years.map((y) => grid[c][y]?.value ?? null),   // null leaves a gap; nothing is interpolated
    facts: years.map((y) => grid[c][y] || null),
    borderColor: COMPANY_COLORS[c],
    backgroundColor: COMPANY_COLORS[c],
    borderWidth: 2,
    pointStyle: POINT_STYLES[c],
    pointRadius: POINT_STYLES[c] === "triangle" ? 6.5 : 5,   // a triangle looks smaller than a circle of the same radius
    pointHoverRadius: 8,
    pointBorderColor: "#ffffff",
    pointBorderWidth: 2,
    pointHitRadius: 12,
    spanGaps: false,
    borderRadius: 4,
    borderSkipped: "start",
    maxBarThickness: 28,
    categoryPercentage: 0.72,
    barPercentage: 0.88,
  }));
  const pickAt = (event, chart) => chart.getElementsAtEventForMode(event, "nearest", { intersect: true }, false)[0];
  return new window.Chart(canvas, {
    type,
    data: { labels: years, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: legendBelow ? "bottom" : "right",
          align: legendBelow ? "center" : "start",
          // Line charts key each company by its point shape; bar charts by a square, like the bars.
          labels: { usePointStyle: type === "line", boxWidth: 10, boxHeight: 10, padding: 16, color: "#344054", font: { size: 13 } },
        },
        tooltip: {
          backgroundColor: "#18212F",
          padding: 10,
          boxPadding: 4,
          usePointStyle: true,
          callbacks: {
            label: (ctx) => {
              const f = ctx.dataset.facts[ctx.dataIndex];
              return f ? ` ${ctx.dataset.label}: ${formatValue(f.value, unit)} (p. ${f.page_no})` : ` ${ctx.dataset.label}: not available`;
            },
          },
        },
        title: { display: false, text: "", align: "start", color: "#18212F", font: { size: 15, weight: "600" } },
        subtitle: { display: false, text: "", align: "start", color: "#5D6679", padding: { bottom: 12 } },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: "#D0D5DD" },
          ticks: { color: "#475467" },
          title: { display: true, text: "Financial year", color: "#5D6679" },
        },
        y: {
          beginAtZero: true,
          grid: { color: "#EAECF0" },
          border: { display: false },
          ticks: { color: "#475467", callback: (v) => (unit === "%" ? `${axisNumber.format(v)}%` : axisNumber.format(v)) },
          title: { display: true, text: unitLabel(unit) === "%" ? "Per cent" : unitLabel(unit), color: "#5D6679" },
        },
      },
      onClick: (event, _elements, chart) => {
        const hit = pickAt(event, chart);
        const fact = hit && chart.data.datasets[hit.datasetIndex].facts[hit.index];
        if (fact) onPick([fact]);
      },
      onHover: (event, _elements, chart) => {
        event.native.target.style.cursor = pickAt(event, chart) ? "pointer" : "default";
      },
    },
  });
}

/** Save the chart as a PNG, with its title and sources drawn in so the file stands on its own. */
export function downloadChart(chart, { title, subtitle, filename }) {
  const { plugins } = chart.options;
  Object.assign(plugins.title, { display: true, text: title });
  Object.assign(plugins.subtitle, { display: true, text: subtitle });
  chart.update("none");
  const link = h("a", { href: chart.toBase64Image("image/png", 1), download: filename });
  plugins.title.display = false;
  plugins.subtitle.display = false;
  chart.update("none");
  link.click();
}

// ---- Table ---------------------------------------------------------------------------------------

/**
 * Companies as rows, years as columns, then the change in the last year and the CAGR.
 * Growth needs two years and CAGR needs two years with a value, so either is shown only where they exist.
 */
export function metricTable({ unit, companies, years, grid, onPick, caption }) {
  const last = years.at(-1);
  const prev = previousFy(last);
  const growthFacts = (c) => (grid[c][prev] && grid[c][last] ? [grid[c][prev], grid[c][last]] : null);
  const showGrowth = companies.some(growthFacts);
  const span = (c) => years.filter((y) => grid[c][y]);
  const showCagr = unit !== "%" && years.length >= 2 && companies.some((c) => span(c).length >= 2);

  const head = h("tr", null,
    h("th", { scope: "col" }, "Company"),
    years.map((y) => h("th", { scope: "col", class: "num", "data-fy": y, title: y }, y.slice(2))),
    showGrowth ? h("th", { scope: "col", class: "num", title: `${prev} to ${last}` },
      unit === "%" ? "Change" : "Growth", h("span", { class: "th-sub" }, `vs ${prev.slice(2)}`)) : "",
    showCagr ? h("th", { scope: "col", class: "num" }, "CAGR") : "");

  const rows = companies.map((c) => {
    const cells = years.map((y) => {
      const f = grid[c][y];
      return h("td", { class: "num" }, f
        ? numberButton(formatNumber(f.value, unit), [f], onPick, `${c} ${y}: ${formatValue(f.value, unit)}, page ${f.page_no}. Show source`)
        : gap());
    });
    let growthCell = "";
    if (showGrowth) {
      const pair = growthFacts(c);
      growthCell = h("td", { class: "num" }, pair
        ? numberButton(formatChange(change(pair[0].value, pair[1].value, unit), unit), pair, onPick,
          `${c}: change from ${prev} to ${last}, from the values on ${pageList(pair)}. Show both sources`)
        : gap());
    }
    let cagrCell = "";
    if (showCagr) {
      const have = span(c);
      const first = grid[c][have[0]];
      const end = grid[c][have.at(-1)];
      const value = have.length >= 2 ? cagr(first.value, end.value, first.fy, end.fy) : null;
      const n = have.length >= 2 ? fyEnd(end.fy) - fyEnd(first.fy) : 0;
      cagrCell = h("td", { class: "num" }, value == null ? gap() : h("span", { class: "cagr" },
        numberButton(formatChange(value, unit).replace("+", ""), [first, end], onPick,
          `${c}: compound annual growth from ${first.fy} to ${end.fy}. Show both sources`),
        h("span", { class: "cagr-span" }, `${n} yr${n === 1 ? "" : "s"}`)));
    }
    return h("tr", null, h("th", { scope: "row" }, h("span", { class: "company-cell" }, swatch(c), c)), cells, growthCell, cagrCell);
  });

  return h("div", { class: "table-wrap" },
    h("table", { class: "data-table metric-table" },
      caption ? h("caption", null, caption) : "",
      h("thead", null, head),
      h("tbody", null, rows)));
}

// ---- Key insights --------------------------------------------------------------------------------

const PHRASE = {
  revenue: "revenue", revenue_usd: "revenue in US$", net_profit: "net profit", operating_margin: "operating / EBIT margin",
  headcount: "headcount", attrition: "attrition",
};

/** Deterministic observations: who is highest, who grew fastest, each year-on-year change, CAGR and gaps. */
export function insights({ metric, unit, companies, years, grid }) {
  const out = [];
  const what = PHRASE[metric] || metricName(metric).toLowerCase();
  const fmt = (f) => formatValue(f.value, unit);
  const latest = [...years].reverse().find((y) => companies.some((c) => grid[c][y]));
  if (!latest) return out;

  const ranked = companies.map((c) => grid[c][latest]).filter(Boolean).sort((a, b) => b.value - a.value);
  if (ranked.length >= 2) {
    const rest = ranked.slice(1).map((f) => `${f.company} (${fmt(f)})`);
    out.push({ text: `${ranked[0].company} has the highest ${what} in ${latest} at ${fmt(ranked[0])}, followed by ${listing(rest)}.`, facts: ranked });
  } else if (ranked.length === 1) {
    out.push({ text: `Only ${ranked[0].company} has a ${what} value for ${latest} in the extracted data: ${fmt(ranked[0])}.`, facts: ranked });
  }

  const prev = previousFy(latest);
  const moves = companies
    .filter((c) => grid[c][prev] && grid[c][latest])
    .map((c) => ({ c, from: grid[c][prev], to: grid[c][latest], delta: change(grid[c][prev].value, grid[c][latest].value, unit) }))
    .filter((m) => m.delta != null)
    .sort((a, b) => b.delta - a.delta);
  if (moves.length >= 2) {
    const top = moves[0];
    const others = moves.slice(1).map((m) => `${m.c} (${formatChange(m.delta, unit)})`);
    const lead = unit === "%"
      ? `Largest change from ${prev} to ${latest}: ${top.c} (${formatChange(top.delta, unit)})`
      : top.delta > 0
        ? `Fastest growth from ${prev} to ${latest}: ${top.c} (${formatChange(top.delta, unit)})`
        : `Every company declined from ${prev} to ${latest}; ${top.c} declined least (${formatChange(top.delta, unit)})`;
    out.push({ text: `${lead}, then ${listing(others)}.`, facts: moves.flatMap((m) => [m.from, m.to]) });
  }
  for (const m of moves) {
    out.push({ text: `${m.c}: ${fmt(m.from)} in ${prev} → ${fmt(m.to)} in ${latest} (${formatChange(m.delta, unit)} year on year).`, facts: [m.from, m.to] });
  }

  if (unit !== "%") {
    for (const c of companies) {
      const have = years.filter((y) => grid[c][y]);
      if (have.length < 3) continue;   // with two years the CAGR is the year-on-year change above
      const first = grid[c][have[0]];
      const end = grid[c][have.at(-1)];
      const rate = cagr(first.value, end.value, first.fy, end.fy);
      if (rate == null) continue;
      const n = fyEnd(end.fy) - fyEnd(first.fy);
      out.push({ text: `${c}'s ${what} grew at a compound ${rate.toFixed(1)}% a year over ${n} years, from ${fmt(first)} in ${first.fy} to ${fmt(end)} in ${end.fy}.`, facts: [first, end] });
    }
  }

  // Say plainly what is missing, grouped by the years that are available.
  const groups = new Map();
  for (const c of companies) {
    const have = years.filter((y) => grid[c][y]);
    if (have.length === years.length) continue;
    const key = have.join(",");
    groups.set(key, [...(groups.get(key) || []), c]);
  }
  for (const [key, names] of groups) {
    const have = key ? key.split(",") : [];
    out.push({
      text: have.length
        ? `${listing(names)}: only ${listing(have)} ${have.length === 1 ? "is" : "are"} in the extracted data for ${what}.`
        : `${listing(names)}: no ${what} value for ${listing(years)} in the extracted data.`,
      facts: [], gap: true,
    });
  }
  return out;
}

function insightList(items, onPick) {
  if (!items.length) return h("p", { class: "muted" }, "No values to compare for this selection.");
  return h("ul", { class: "insights" }, items.map((item) => h("li", { class: item.gap ? "is-gap" : null },
    h("span", null, item.text),
    item.facts.length ? h("span", { class: "insight-sources" }, sourceChips(item.facts, onPick, true)) : "")));
}

/** One chip per report page used, e.g. "TCS · p. 167". */
export function sourceChips(facts, onPick, small = false) {
  const pages = new Map();
  for (const f of facts) {
    const key = `${f.doc_id}:${f.page_no}`;
    pages.set(key, [...(pages.get(key) || []), f]);
  }
  return [...pages.values()].map((group) => {
    const f = group[0];
    const years = [...new Set(group.map((g) => g.fy))].sort();
    return h("button", {
      type: "button", class: small ? "source-chip small" : "source-chip",
      title: `${f.report}, page ${f.page_no} (${years.join(", ")})`,
      "aria-label": `Source: ${f.company}, page ${f.page_no}, ${years.join(", ")}`,
      onclick: () => onPick(group),
    }, icon("pdf", "icon pdf-icon"), `${f.company} · p. ${f.page_no}`);
  });
}

// ---- The metric panel ----------------------------------------------------------------------------

let panelCount = 0;

/**
 * A card with metric and chart-type selectors, a download button, and the chart, table and key insights.
 * layout 'tabs' (inside an answer) shows one of the three at a time; 'stacked' (Compare view) shows all three.
 * onPick(facts, metric) is called when a number, point or source chip is selected.
 */
export function metricPanel({ metric, companies, year = "all", layout = "stacked", onPick, onMetricChange, highlightFy = null,
                              chartType = null }) {
  const id = `mp${++panelCount}`;
  let current = metric;
  let type = "line";       // what is drawn: 'line' | 'bar'
  // the type the question asked for ('bar graph of headcount'), then whatever the user picks; a pie is drawn as bars
  let chosenType = chartType === "bar" || chartType === "line" ? chartType : chartType === "pie" ? "bar" : null;
  let chart = null;
  let shown = { grid: null, years: [], unit: "" };   // what the chart and table currently show
  let loads = 0;   // only the latest metric request may render

  const metricSelect = h("select", { id: `${id}-metric`, onchange: () => { current = metricSelect.value; onMetricChange?.(current); load(); } },
    state.metrics.map((m) => h("option", { value: m.metric, selected: m.metric === current }, metricName(m.metric))));
  const typeSelect = h("select", { id: `${id}-type`, onchange: () => { chosenType = type = typeSelect.value; redrawChart(); } },
    h("option", { value: "line" }, "Line"), h("option", { value: "bar" }, "Bar"));
  const download = h("button", { type: "button", class: "btn btn-outline", onclick: () => saveChart() }, icon("download"), "Download chart");

  const title = h("h3", { class: "mp-title" });
  const chartBox = h("div", { class: "mp-chart" });
  const chartPane = h("div", { class: "mp-pane", id: `${id}-chart`, role: layout === "tabs" ? "tabpanel" : null, "aria-labelledby": layout === "tabs" ? `${id}-tab-chart` : null }, chartBox);
  const tablePane = h("div", { class: "mp-pane", id: `${id}-table`, role: layout === "tabs" ? "tabpanel" : null, "aria-labelledby": layout === "tabs" ? `${id}-tab-table` : null });
  const insightPane = h("div", { class: "mp-pane", id: `${id}-insights`, role: layout === "tabs" ? "tabpanel" : null, "aria-labelledby": layout === "tabs" ? `${id}-tab-insights` : null });
  const sources = h("div", { class: "mp-sources" });
  const status = h("div", { class: "mp-status" });

  const panes = { chart: chartPane, table: tablePane, insights: insightPane };
  let tabs = null;
  if (layout === "tabs") {
    tabs = h("div", { class: "tabs mp-tabs", role: "tablist", "aria-label": "Answer views" },
      [["chart", "Chart"], ["table", "Table"], ["insights", "Key insights"]].map(([key, label]) => h("button", {
        type: "button", role: "tab", id: `${id}-tab-${key}`, class: "tab", "aria-controls": `${id}-${key}`,
        "aria-selected": String(key === "chart"), tabindex: key === "chart" ? "0" : "-1",
        onclick: (event) => selectTab(key, event.currentTarget),
      }, label)));
    tabKeys(tabs);
    tablePane.hidden = true;
    insightPane.hidden = true;
  }

  function selectTab(key, button) {
    for (const b of tabs.querySelectorAll('[role="tab"]')) {
      const on = b === button;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    for (const [k, pane] of Object.entries(panes)) pane.hidden = k !== key;
    download.hidden = key !== "chart";
    if (key === "chart") chart?.resize();
  }

  const controls = h("div", { class: "mp-controls" },
    h("label", { class: "inline-field", for: `${id}-type` }, "Chart type"), typeSelect,
    h("label", { class: "inline-field", for: `${id}-metric` }, "Metric"), metricSelect);

  const root = h("section", { class: `card metric-panel layout-${layout}`, "aria-label": "Metric comparison" },
    h("div", { class: "mp-head" }, tabs || h("span"), controls),
    h("div", { class: "mp-title-row" }, title, download),
    status,
    layout === "tabs"
      ? h("div", { class: "mp-body" }, chartPane, tablePane, insightPane)
      : h("div", { class: "mp-body" }, chartPane,
        h("h4", { class: "mp-subhead" }, "Table"), tablePane,
        h("h4", { class: "mp-subhead" }, icon("bulb"), "Key insights"), insightPane),
    sources);

  async function load() {
    const run = ++loads;
    root.classList.add("is-loading");   // the previous chart stays, dimmed, until the new one is ready
    status.replaceChildren();
    let data;
    try {
      data = await api.metrics(current);
    } catch (err) {
      if (run !== loads) return;
      root.classList.remove("is-loading");
      chart?.destroy();
      chart = null;
      for (const box of [chartBox, tablePane, insightPane, sources]) box.replaceChildren();
      status.replaceChildren(stateBox("error", "Could not load this metric", err.message,
        h("button", { type: "button", class: "btn btn-ghost btn-sm", onclick: load }, "Try again")));
      return;
    }
    if (run !== loads) return;
    root.classList.remove("is-loading");
    const unit = data.unit || metricInfo(current).unit;
    const grid = gridOf(data.facts, companies);
    const years = year === "all" ? yearsOf(grid) : [year];
    shown = { grid, years, unit };
    const name = metricName(current);
    title.textContent = `${name} (${unitLabel(unit)})`;
    const pick = (facts) => onPick(facts, current);
    const used = years.flatMap((y) => companies.map((c) => grid[c][y]).filter(Boolean));

    if (!used.length) {
      chart?.destroy();
      chart = null;
      chartBox.replaceChildren(stateBox("empty", `No ${name.toLowerCase()} values for ${listing(years)}`,
        `${GAP_TEXT} for ${listing(companies)}.`));
    }
    tablePane.replaceChildren(metricTable({ unit, companies, years, grid, onPick: pick }),
      notes(current, unit, companies, years, grid));
    if (highlightFy) {
      tablePane.querySelector(`thead th[data-fy="${highlightFy}"]`)?.classList.add("hl-col");
    }
    insightPane.replaceChildren(insightList(insights({ metric: current, unit, companies, years, grid }), pick));
    sources.replaceChildren(...(used.length ? [h("span", { class: "mp-sources-label" }, "Sources:"), ...sourceChips(used, pick)] : []));

    // One year cannot make a line; two years read better as grouped bars. The user's choice wins after that.
    typeSelect.querySelector('[value="line"]').disabled = years.length < 2;
    type = chosenType && !(chosenType === "line" && years.length < 2) ? chosenType : years.length >= 3 ? "line" : "bar";
    typeSelect.value = type;
    download.disabled = !used.length;
    if (used.length) redrawChart();
  }

  function redrawChart() {
    chart?.destroy();
    const canvas = h("canvas", {
      role: "img",
      "aria-label": `${type === "line" ? "Line" : "Bar"} chart of ${title.textContent} for ${listing(companies)}, ${listing(shown.years)}. The table lists every value.`,
    });
    chartBox.replaceChildren(h("div", { class: "chart-frame" }, canvas), notes(current, shown.unit, companies, shown.years, shown.grid, true));
    chart = drawChart(canvas, {
      type, unit: shown.unit, companies, years: shown.years, grid: shown.grid,
      onPick: (facts) => onPick(facts, current), legendBelow: root.clientWidth < 600,
    });
  }

  function saveChart() {
    if (!chart) return;
    const used = shown.years.flatMap((y) => companies.map((c) => shown.grid[c][y]).filter(Boolean));
    const pages = companies.map((c) => {
      const p = [...new Set(used.filter((f) => f.company === c).map((f) => f.page_no))];
      return p.length ? `${c} p. ${p.join(", ")}` : null;
    }).filter(Boolean);
    downloadChart(chart, {
      title: title.textContent,
      subtitle: `Source: FY2025-26 annual reports (${pages.join("; ")})${current === "operating_margin" ? `. ${MARGIN_NOTE}` : ""}`,
      filename: `${current}-${companies.join("-").toLowerCase()}.png`,
    });
  }

  load();
  return { root, destroy: () => chart?.destroy() };
}

/** Notes under a chart or table: the margin definition, and which values are missing. */
function notes(metric, unit, companies, years, grid, forChart = false) {
  const lines = [];
  if (metric === "operating_margin") lines.push(MARGIN_NOTE);
  if (forChart) {
    const missing = companies.filter((c) => years.some((y) => !grid[c][y]));
    if (missing.length) lines.push(`Gaps are years not available in the extracted data (${listing(missing)}); nothing is filled in.`);
  } else if (companies.some((c) => years.some((y) => !grid[c][y]))) {
    lines.push("— marks a value not available in the extracted data. Select any number to see its source page.");
  } else {
    lines.push("Select any number to see its source page.");
  }
  return h("div", { class: "notes" }, lines.map((line) => h("p", { class: "note" }, icon("info"), line)));
}
