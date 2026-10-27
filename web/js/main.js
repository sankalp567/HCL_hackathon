// Entry point: the top-bar filters, hash routing between views (#ask, #dashboard, ...), and start-up.
// URL parameters: ?q=question asks it on load; ?year=all|FY2024-25 and ?companies=TCS,Infosys set the filters.
import { api } from "./api.js";
import { ask, initChat, renderWelcome, updateScopeHint } from "./chat.js";
import { setPanelView } from "./evidence.js";
import { onFiltersChange, setFilters, state } from "./state.js";
import { COMPANIES, hydrateIcons } from "./util.js";
import { initExplorer, renderCompare, renderDashboard, renderData, renderExplorer, renderSettings } from "./views.js";

const VIEWS = {
  ask: { title: "Ask AI", render: () => {} },
  dashboard: { title: "Dashboard", render: renderDashboard },
  compare: { title: "Compare companies", render: renderCompare },
  data: { title: "Extracted data", render: renderData },
  explorer: { title: "Evidence explorer", render: renderExplorer },
  settings: { title: "Settings", render: renderSettings },
};
let current = null;

function route(moveFocus) {
  const name = window.location.hash.slice(1);
  const view = VIEWS[name] ? name : "ask";
  if (view === current) return;
  current = view;
  for (const section of document.querySelectorAll(".view")) section.hidden = section.dataset.view !== view;
  for (const link of document.querySelectorAll(".nav-item")) {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.title = `${VIEWS[view].title} · Annual Report AI`;
  setPanelView(view);
  VIEWS[view].render();
  if (moveFocus) {   // after a navigation, put keyboard and screen-reader focus on the new view
    const target = view === "ask" ? document.getElementById("question") : document.querySelector(`#view-${view} h1`);
    target?.focus({ preventScroll: true });
  }
}

function initFilters() {
  for (const button of document.querySelectorAll(".seg")) {
    button.addEventListener("click", () => {
      const company = button.dataset.company;
      const on = state.companies.includes(company);
      if (on && state.companies.length === 1) return;   // at least one company stays selected
      const next = on ? state.companies.filter((c) => c !== company) : [...state.companies, company];
      setFilters({ companies: COMPANIES.filter((c) => next.includes(c)) });
    });
  }
  document.getElementById("year-filter").addEventListener("change", (event) => setFilters({ year: event.target.value }));
  onFiltersChange(() => {
    syncFilterControls();
    updateScopeHint();
    VIEWS[current].render();
  });
  syncFilterControls();
}

function syncFilterControls() {
  for (const button of document.querySelectorAll(".seg")) {
    const on = state.companies.includes(button.dataset.company);
    button.setAttribute("aria-pressed", String(on));
    button.title = on && state.companies.length === 1 ? "At least one company stays selected" : "";
  }
  document.getElementById("year-filter").value = state.year;
}

async function loadOverview() {
  const pill = document.getElementById("reports-pill");
  const text = document.getElementById("reports-pill-text");
  try {
    const overview = await api.overview();
    state.overview = overview;
    if (overview.metrics?.length) state.metrics = overview.metrics;
    text.textContent = `${overview.reports.length} reports loaded`;
    pill.title = overview.reports.map((r) => `${r.title} (${r.pages} pages)`).join("\n");
    renderWelcome(overview.reports);
  } catch {
    try {
      const health = await api.health();
      text.textContent = `${health.documents ?? 0} reports loaded`;
    } catch {
      text.textContent = "Server not reachable";
      pill.classList.add("pill-error");
    }
  }
}

function readUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const year = params.get("year");
  if (year === "all" || /^FY\d{4}-\d{2}$/.test(year || "")) state.year = year;
  const companies = (params.get("companies") || "").split(",").filter((c) => COMPANIES.includes(c));
  if (companies.length) state.companies = COMPANIES.filter((c) => companies.includes(c));
  return params.get("q");
}

async function start() {
  hydrateIcons();
  for (const h1 of document.querySelectorAll(".view h1")) h1.tabIndex = -1;
  const question = readUrlParams();
  initFilters();
  initChat();
  initExplorer();
  await loadOverview();   // metric names and report titles, used by every view
  window.addEventListener("hashchange", () => route(true));
  if (question && window.location.hash && window.location.hash !== "#ask") window.location.hash = "#ask";
  route(false);
  if (question) ask(question);
}

start();
