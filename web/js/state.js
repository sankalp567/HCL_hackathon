// App-wide state: the company and year filters from the top bar, and what /api/overview reported.
import { COMPANIES, FALLBACK_METRICS } from "./util.js";

export const state = {
  companies: [...COMPANIES],   // selected companies, always in canonical order
  year: "FY2025-26",           // 'FY2025-26' | 'FY2024-25' | 'all'
  metrics: FALLBACK_METRICS,   // [{metric, label, unit}], replaced by /api/overview
  overview: null,
};

const listeners = new Set();

export function onFiltersChange(fn) {
  listeners.add(fn);
}

export function setFilters(patch) {
  Object.assign(state, patch);
  for (const fn of listeners) fn();
}

export const allCompaniesSelected = () => state.companies.length === COMPANIES.length;

export const metricInfo = (metric) =>
  state.metrics.find((m) => m.metric === metric) || { metric, label: metric, unit: "" };
