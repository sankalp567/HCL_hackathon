// The "Evidence & Sources" panel on the right, and the modal that shows a full report page.
// Each view keeps its own list of evidence; the panel shows the list of the view on screen.
import { api, pageImageUrl, pdfUrl } from "./api.js";
import { COMPANIES, formatValue, h, icon, metricName, prefersReducedMotion } from "./util.js";

const $ = (id) => document.getElementById(id);
const panelBody = $("panel-body");
const tabsBox = $("panel-tabs");
const toggle = $("panel-toggle");
const narrow = window.matchMedia("(max-width: 1199px)");

const EMPTY = {
  ask: "Sources for an answer appear here: the report, the page, and the table row or passage behind each claim.",
  dashboard: "Select any figure on the dashboard to see the report page it came from.",
  compare: "Select a chart point, a table value or a source chip to see where the number came from.",
  data: "Select a row to see the table it was read from.",
  explorer: "Search the reports, then select a result to see it here.",
  settings: "Evidence cards appear here when you select a number or a source in any view.",
};

const contexts = new Map();   // view -> {items, empty}
let activeView = "ask";
let activeTab = "All";

// ---- Turning API objects into panel items -------------------------------------------------------

/** A fact from /api/metrics or /api/facts. */
export function factRef(fact, metric = fact.metric) {
  return {
    key: `f${fact.fact_id}`, fact_id: fact.fact_id, company: fact.company, report: fact.report, pdf: fact.pdf,
    doc_id: fact.doc_id, page: fact.page_no, kind: "fact", quote: fact.quote, bbox: fact.bbox, status: fact.status,
    label: `${metricName(metric)} · ${fact.fy} · ${formatValue(fact.value, fact.unit)}`,
  };
}

/** A citation from the answer stream ([n] in the answer text). */
export function citationRef(c) {
  const key = c.fact_id != null ? `f${c.fact_id}` : c.passage_id != null ? `p${c.passage_id}` : `c${c.n}`;
  return { ...c, key };
}

/** A retrieved source from the `meta` event, before the citations arrive. */
export const sourceRef = (s) => ({ ...s, key: `s${s.n}` });

/** A passage from /api/search. */
export function passageRef(r) {
  return {
    key: `p${r.passage_id}`, passage_id: r.passage_id, company: r.company, report: r.report, pdf: r.pdf,
    doc_id: r.doc_id, page: r.page, kind: r.kind, quote: r.snippet.replace(/[[\]]/g, ""), bbox: null, label: r.section,
  };
}

// ---- Panel ---------------------------------------------------------------------------------------

/**
 * Show items in the panel for a view.
 * focus: key of a card to scroll to; append: add to the view's list instead of replacing it;
 * empty: text for an empty list; open: also open the panel if it is hidden.
 */
export function showEvidence(items, { view = activeView, focus = null, append = false, empty = null, open = false } = {}) {
  const previous = contexts.get(view);
  let list = items;
  if (append && previous) {
    const known = new Set(previous.items.map((i) => i.key));
    list = [...previous.items, ...items.filter((i) => !known.has(i.key))];
  }
  contexts.set(view, { items: list, empty: empty ?? (append ? previous?.empty : null) });
  if (view !== activeView) return;
  // A new list starts on "All"; an addition keeps the company tab unless that tab would hide the card to focus.
  const hidden = focus && activeTab !== "All" && !list.some((i) => i.key === focus && i.company === activeTab);
  if (!append || hidden) activeTab = "All";
  render();
  if (open) openPanel();
  if (focus) focusCard(focus);
}

export function setPanelView(view) {
  activeView = view;
  activeTab = "All";
  render();
}

function render() {
  const { items, empty } = contexts.get(activeView) || { items: [], empty: null };
  $("panel-count").textContent = items.length;

  tabsBox.replaceChildren(...["All", ...COMPANIES].map((name) => {
    const count = name === "All" ? items.length : items.filter((i) => i.company === name).length;
    const selected = name === activeTab;
    return h("button", {
      type: "button", role: "tab", class: "tab", "aria-selected": String(selected), tabindex: selected ? "0" : "-1",
      onclick: () => { activeTab = name; render(); tabsBox.querySelector('[aria-selected="true"]').focus(); },
    }, name === "All" ? "All sources" : name, h("span", { class: "tab-count" }, ` (${count})`));
  }));

  const shown = activeTab === "All" ? items : items.filter((i) => i.company === activeTab);
  if (!items.length) panelBody.replaceChildren(emptyNote(empty || EMPTY[activeView]));
  else if (!shown.length) panelBody.replaceChildren(emptyNote(`No ${activeTab} sources here.`));
  else panelBody.replaceChildren(...shown.map(card));
}

const emptyNote = (text) => h("div", { class: "panel-empty" }, icon("layers"), h("p", null, text));

// Arrow keys move between tabs, as in any tab list.
export function tabKeys(list) {
  list.addEventListener("keydown", (event) => {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const at = tabs.indexOf(document.activeElement);
    if (at < 0) return;
    const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[event.key];
    if (next == null) return;
    event.preventDefault();
    tabs[(next + tabs.length) % tabs.length].click();
  });
}
tabKeys(tabsBox);

function card(item) {
  const content = h("div", { class: "ev-content" }, h("div", { class: "skeleton" }), h("div", { class: "skeleton short" }));
  const collapse = h("button", {
    type: "button", class: "icon-btn ev-collapse", "aria-expanded": "true", "aria-label": "Collapse this source",
    onclick: () => {
      const open = collapse.getAttribute("aria-expanded") === "true";
      collapse.setAttribute("aria-expanded", String(!open));
      collapse.setAttribute("aria-label", open ? "Expand this source" : "Collapse this source");
      el.classList.toggle("collapsed", open);
    },
  }, icon("chevron"));

  const label = h("p", { class: "ev-label", hidden: !item.label }, item.label || "");
  const el = h("article", { class: "ev-card", "data-key": item.key, tabindex: "-1", "aria-label": `${item.company}, page ${item.page}` },
    h("header", { class: "ev-head" },
      icon("pdf", "icon pdf-icon"),
      h("p", { class: "ev-report" },
        item.n != null ? h("span", { class: "ev-n" }, `[${item.n}]`) : null,
        h("span", { class: "ev-company" }, item.company), ` · ${reportTitle(item)}`),
      h("span", { class: "badge" }, `Page ${item.page}`),
      collapse),
    label,
    content,
    h("footer", { class: "ev-foot" },
      trust(item),
      h("button", { type: "button", class: "btn btn-outline btn-sm", onclick: () => openPage(item) }, "View full page", icon("external"))));
  fill(item, content, label);
  return el;
}

/** "TCS Annual Report 2025-26" shown after "TCS · " would repeat the company, so drop it from the title. */
export function reportTitle(item) {
  const title = (item.report || "").replace(/(\d{4})-(\d{2})\b/g, "$1\u2011$2");   // keep 2025-26 on one line
  return title.startsWith(`${item.company} `) ? title.slice(item.company.length + 1) : title;
}

// Facts and passages have a richer view (the table row, the section) behind /api/evidence.
async function fill(item, box, label) {
  if (item.fact_id == null && item.passage_id == null) {
    box.replaceChildren(quote(item.quote, false));
    return;
  }
  try {
    const ev = await api.evidence(item);
    if (!item.label && ev.metric) {   // a cited fact: name the number it supports
      label.textContent = `${metricName(ev.metric)} · ${ev.fy} · ${formatValue(ev.value, ev.unit)}`;
      label.hidden = false;
    }
    box.replaceChildren(
      ev.title ? h("p", { class: "ev-section" }, ev.title) : "",
      ev.table?.rows?.length ? miniTable(ev.table) : "",
      quote(ev.quote || item.quote, Boolean(ev.table?.rows?.length)));
  } catch (err) {
    box.replaceChildren(quote(item.quote, false), h("p", { class: "ev-error" }, `Could not load the table view (${err.message})`));
  }
}

function quote(text, belowTable) {
  if (!text) return "";
  const p = h("blockquote", { class: belowTable ? "ev-quote small" : "ev-quote" }, `“${text}”`);
  if (text.length < 260) return p;
  // Long passages start clamped to a few lines.
  p.classList.add("clamped");
  const more = h("button", { type: "button", class: "link-btn", "aria-expanded": "false",
    onclick: () => {
      const clamped = p.classList.toggle("clamped");
      more.textContent = clamped ? "Show more" : "Show less";
      more.setAttribute("aria-expanded", String(!clamped));
    } }, "Show more");
  return h("div", null, p, more);
}

const MAX_ROWS = 8;

function miniTable(t) {
  let rows = t.rows;
  if (rows.length > MAX_ROWS) {   // keep the window around the highlighted row
    const at = Math.max(0, rows.findIndex((r) => r.highlight));
    const start = Math.max(0, Math.min(at - 2, rows.length - MAX_ROWS));
    rows = rows.slice(start, start + MAX_ROWS);
  }
  const hasHeader = (t.columns || []).some((c) => c);
  const cellClass = (row, i) => [i === t.highlight_col ? "hl-col" : "", row.highlight && i === t.highlight_col ? "hl-cell" : ""]
    .join(" ").trim() || null;
  return h("div", { class: "mini-table-wrap" },
    h("table", { class: "mini-table" },
      hasHeader ? h("thead", null, h("tr", null, t.columns.map((c, i) => h("th", { scope: "col", class: i === t.highlight_col ? "hl-col" : null }, c)))) : "",
      h("tbody", null, rows.map((row) => h("tr", { class: row.highlight ? "hl-row" : null },
        row.cells.map((c, i) => i === 0 ? h("th", { scope: "row" }, c) : h("td", { class: cellClass(row, i) }, c)))))));
}

const STATUS_TEXT = {
  verified: "Verified",
  corroborated: "Corroborated by another table in the report",
  single_source: "Found in one place in the report",
  conflict: "The report prints a different value elsewhere",
  not_in_text: "Not on the page's text layer; check the image",
};

function trust(item) {
  if (item.verified === true) return h("span", { class: "trust ok" }, icon("check"), "Numbers match the source");
  if (item.verified === false) return h("span", { class: "trust" }, icon("info"), "Check the page: numbers not matched");
  if (item.status) {
    const good = item.status === "corroborated" || item.status === "verified";
    return h("span", { class: good ? "trust ok" : "trust" }, icon(good ? "check" : "info"), STATUS_TEXT[item.status] || item.status);
  }
  return h("span");
}

function focusCard(key) {
  const el = panelBody.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (!el) return;
  el.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  el.classList.remove("flash");
  void el.offsetWidth;   // restart the highlight animation
  el.classList.add("flash");
  el.focus({ preventScroll: true });
}

// Wide screens: the panel is a column that can be collapsed. Narrow screens: it is a drawer.
const isOpen = () => narrow.matches
  ? document.body.classList.contains("panel-open")
  : !document.body.classList.contains("panel-collapsed");

export function openPanel() {
  document.body.classList.add("panel-open");
  document.body.classList.remove("panel-collapsed");
  syncToggle();
}

function closePanel() {
  document.body.classList.remove("panel-open");
  if (!narrow.matches) document.body.classList.add("panel-collapsed");
  syncToggle();
  toggle.focus();
}

function syncToggle() {
  toggle.setAttribute("aria-expanded", String(isOpen()));
}

toggle.addEventListener("click", () => (isOpen() ? closePanel() : openPanel()));
$("panel-close").addEventListener("click", closePanel);
$("panel").addEventListener("keydown", (event) => {
  if (event.key === "Escape" && narrow.matches) closePanel();
});
narrow.addEventListener("change", () => {
  document.body.classList.remove("panel-open");
  syncToggle();
});
syncToggle();

// ---- Full-page modal ----------------------------------------------------------------------------

const modal = $("page-modal");
const image = $("modal-image");
const figure = $("modal-figure");
const loading = $("modal-loading");

let openedFrom = null;   // focus returns here when the modal closes

/** Show the page behind an item. returnTo gets focus back when the modal closes (default: the focused element). */
export function openPage(item, returnTo = null) {
  if (!modal.open) openedFrom = returnTo || document.activeElement;
  $("modal-title").textContent = `${item.company} · ${reportTitle(item)}`;
  $("modal-page").textContent = `Page ${item.page}`;
  $("modal-quote").textContent = item.quote ? `“${item.quote}”` : "";
  $("modal-pdf").href = pdfUrl(item.pdf, item.page);
  $("modal-pdf-text").textContent = `Open PDF at page ${item.page}`;
  image.alt = `Page ${item.page} of the ${item.report}${item.bbox ? ", with the cited value highlighted" : ""}`;
  image.hidden = true;
  loading.textContent = "Rendering the page…";
  loading.hidden = false;
  image.onload = () => {
    loading.hidden = true;
    image.hidden = false;
    // The page is rendered at 2x; scroll so the highlighted box sits in the upper third of the view.
    if (item.bbox) figure.scrollTop = (item.bbox[1] * 2 * image.clientHeight) / image.naturalHeight - figure.clientHeight / 3;
  };
  image.onerror = () => { loading.textContent = "This page could not be rendered. Open the PDF instead."; };
  image.src = pageImageUrl(item.doc_id, item.page, item.bbox);
  figure.scrollTop = 0;
  if (!modal.open) modal.showModal();
}

$("modal-close").addEventListener("click", () => modal.close());
modal.addEventListener("close", () => {
  image.removeAttribute("src");
  openedFrom?.focus?.();
});
modal.addEventListener("click", (event) => { if (event.target === modal) modal.close(); });   // backdrop click
