// Shared helpers: companies and their colours, metric names, number formatting, and a small DOM builder.

export const COMPANIES = ["TCS", "Infosys", "HCLTech"];

// Validated as a set (colour-blind separation and >= 3:1 against white): blue, orange, green.
export const COMPANY_COLORS = { TCS: "#2a78d6", Infosys: "#ea6a1f", HCLTech: "#12a36f" };
// A second channel besides colour, so lines stay distinguishable without it.
export const POINT_STYLES = { TCS: "circle", Infosys: "rectRounded", HCLTech: "triangle" };

export const FONT_STACK = 'Inter, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial, sans-serif';
export const GAP_TEXT = "Not available in the extracted data";
export const MARGIN_NOTE = "HCLTech reports EBIT margin; TCS and Infosys report operating margin.";

// Short names for selectors and headings; the API's own label is shown where space allows.
export const METRIC_NAMES = {
  revenue: "Revenue",
  revenue_usd: "Revenue in US$",
  net_profit: "Net profit",
  operating_margin: "Operating / EBIT margin",
  headcount: "Headcount",
  attrition: "Attrition",
};

// Used only if /api/overview cannot be reached; the numbers always come from the API.
export const FALLBACK_METRICS = [
  { metric: "revenue", label: "Revenue from operations", unit: "INR crore" },
  { metric: "revenue_usd", label: "Revenue", unit: "USD million" },
  { metric: "net_profit", label: "Net profit attributable to shareholders", unit: "INR crore" },
  { metric: "operating_margin", label: "Operating margin (HCLTech reports EBIT margin)", unit: "%" },
  { metric: "headcount", label: "Employees", unit: "count" },
  { metric: "attrition", label: "Attrition", unit: "%" },
];

export const metricName = (metric) => METRIC_NAMES[metric] || metric.replace(/_/g, " ");

export function unitLabel(unit) {
  return { "INR crore": "₹ crore", "USD million": "US$ million", "%": "%", count: "employees" }[unit] || unit || "";
}

const whole = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

// Percentages keep the decimals the report printed (25.0%, 12.51%).
function percent(value) {
  const decimals = Math.min(2, Math.max(1, (String(value).split(".")[1] || "").length));
  return `${value.toFixed(decimals)}%`;
}

/** The bare number in Indian grouping (2,67,021), for tables whose header names the unit. */
export function formatNumber(value, unit) {
  if (value == null) return "—";
  return unit === "%" ? percent(value) : whole.format(value);
}

/** The number with its unit (₹2,67,021 crore, 25.0%, 6,17,437 employees). */
export function formatValue(value, unit) {
  if (value == null) return "—";
  switch (unit) {
    case "INR crore": return `₹${whole.format(value)} crore`;
    case "USD million": return `US$${whole.format(value)} million`;
    case "%": return percent(value);
    case "count": return `${whole.format(value)} employees`;
    default: return `${whole.format(value)} ${unit || ""}`.trim();
  }
}

/** A signed change: relative % for amounts, percentage points for metrics that are already percentages. */
export function formatChange(change, unit) {
  if (change == null || !Number.isFinite(change)) return "—";
  const sign = change > 0 ? "+" : change < 0 ? "−" : "";
  return unit === "%" ? `${sign}${Math.abs(change).toFixed(1)} pp` : `${sign}${Math.abs(change).toFixed(1)}%`;
}

// Fiscal years look like 'FY2025-26'; the year it ends in is 2026.
export const fyEnd = (fy) => Number(fy.slice(2, 6)) + 1;
export function previousFy(fy) {
  const start = Number(fy.slice(2, 6));
  return `FY${start - 1}-${String(start % 100).padStart(2, "0")}`;
}

/** Change from a to b: relative % for amounts, difference in points for percentages. */
export function change(a, b, unit) {
  if (a == null || b == null) return null;
  if (unit === "%") return b - a;
  return a === 0 ? null : ((b - a) / Math.abs(a)) * 100;
}

/** Compound annual growth between two fiscal years, in %. */
export function cagr(first, last, firstFy, lastFy) {
  const years = fyEnd(lastFy) - fyEnd(firstFy);
  if (years < 1 || first <= 0 || last <= 0) return null;
  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

/**
 * Build an element: h("button", { class: "chip", onclick: fn }, "text", child).
 * Text is always inserted as text nodes, so report content can never inject markup.
 */
export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Icons are fixed SVG strings written here (never data), drawn with the current text colour.
const ICONS = {
  chat: '<path d="M20 15.5a2 2 0 0 1-2 2H8l-4 3.5V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z"/><path d="M8.5 9.5h7M8.5 13h4.5"/>',
  grid: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/>',
  bars: '<path d="M4 20h16"/><path d="M7 16.5V11M12 16.5V6M17 16.5v-3.5"/>',
  table: '<rect x="3.5" y="4" width="17" height="16" rx="2"/><path d="M3.5 9h17M3.5 14.5h17M9.5 9v11"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  sliders: '<path d="M5 20v-6M5 10V4M12 20v-8M12 8V4M19 20v-4M19 12V4"/><path d="M3 14h4M10 8h4M17 16h4"/>',
  pdf: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M8.5 16.5h7M8.5 13h4"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="m8.2 12.3 2.6 2.6 5-5.4"/>',
  tick: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  external: '<path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"/>',
  download: '<path d="M12 4v11"/><path d="m7.5 10.5 4.5 4.5 4.5-4.5"/><path d="M5 20h14"/>',
  send: '<path d="M21 3 3 10.5l7 3 3 7z"/><path d="M21 3 10 13.5"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  bot: '<rect x="4" y="8" width="16" height="11.5" rx="3"/><path d="M12 8V4.5"/><circle cx="12" cy="3.6" r="1.1"/><path d="M9 13.2v.6M15 13.2v.6"/><path d="M2 12.5v3M22 12.5v3"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4.5 20.5a7.5 7.5 0 0 1 15 0"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.4"/>',
  alert: '<path d="M10.3 4.2 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4M12 16.8v.2"/>',
  ban: '<circle cx="12" cy="12" r="9"/><path d="m5.7 5.7 12.6 12.6"/>',
  bulb: '<path d="M9.5 18h5M10.5 21h3"/><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z"/>',
  layers: '<path d="m12 3 9 4.5-9 4.5-9-4.5z"/><path d="m3 12 9 4.5 9-4.5M3 16.5 12 21l9-4.5"/>',
  sort: '<path d="m8 9 4-4 4 4M8 15l4 4 4-4"/>',
  star: '<path d="m12 4 2.4 4.9 5.4.8-3.9 3.8.9 5.4L12 16.4l-4.8 2.5.9-5.4-3.9-3.8 5.4-.8z"/>',
};

export function icon(name, className = "icon") {
  const span = document.createElement("span");
  span.className = className;
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
  return span;
}

/** Fill every <span data-icon="name"> placeholder written in index.html. */
export function hydrateIcons(root = document) {
  for (const slot of root.querySelectorAll("[data-icon]")) {
    slot.replaceWith(icon(slot.dataset.icon, `icon ${slot.className}`.trim()));
  }
}

/** A coloured dot that marks a company; the company name always sits next to it as text. */
export const swatch = (company) => h("span", { class: "swatch", "data-company": company, "aria-hidden": "true" });

export const spinner = () => h("span", { class: "spinner", "aria-hidden": "true" });

export const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** A message box for empty, loading and error states. */
export function stateBox(kind, title, text, action) {
  const mark = kind === "loading" ? spinner() : icon(kind === "error" ? "alert" : "info");
  return h("div", { class: `state-box state-${kind}`, role: kind === "error" ? "alert" : null },
    mark,
    h("div", null, h("p", { class: "state-title" }, title), text ? h("p", { class: "state-text" }, text) : null, action || null));
}
