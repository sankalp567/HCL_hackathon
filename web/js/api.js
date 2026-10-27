// Every call to the backend. Failures become ApiError with a message that can be shown as is.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

async function getJSON(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") url.searchParams.set(key, value);
  }
  let res;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError("Could not reach the server. Check that it is running, then try again.");
  }
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch { /* body was not JSON */ }
    if (res.status === 404 && !detail) throw new ApiError(`${path} is not available on this server.`, 404);
    throw new ApiError(`The server answered ${res.status}${detail ? `: ${detail}` : ""}.`, res.status);
  }
  return res.json();
}

// Responses that do not change while the app is open are fetched once. A failed request is not cached.
const cache = new Map();
function once(key, load) {
  if (!cache.has(key)) cache.set(key, load().catch((err) => { cache.delete(key); throw err; }));
  return cache.get(key);
}

export const api = {
  health: () => getJSON("/api/health"),
  overview: () => once("overview", () => getJSON("/api/overview")),
  /** {metric, label, unit, facts}: one chosen value per company × year. */
  metrics: (metric) => once(`metrics:${metric}`, () => getJSON("/api/metrics", { metric })),
  facts: () => once("facts", () => getJSON("/api/facts")),
  /** The table row or passage behind a fact or a passage. */
  evidence: ({ fact_id, passage_id }) => fact_id != null
    ? once(`fact:${fact_id}`, () => getJSON("/api/evidence", { fact_id }))
    : once(`passage:${passage_id}`, () => getJSON("/api/evidence", { passage_id })),
  search: (q, company) => getJSON("/api/search", { q, company }),
};

export function pageImageUrl(docId, page, bbox) {
  const box = bbox && bbox.every((v) => v != null) ? `?x0=${bbox[0]}&y0=${bbox[1]}&x1=${bbox[2]}&y1=${bbox[3]}` : "";
  return `/api/page/${encodeURIComponent(docId)}/${page}.png${box}`;
}

export const pdfUrl = (pdf, page) => `/reports/${encodeURIComponent(pdf)}#page=${page}`;

/**
 * POST /api/ask and call onEvent(event, data) for each server-sent event as it arrives:
 * meta, token, citations, notice, error, done (see app/main.py for the contract).
 */
export async function askStream(question, companies, onEvent, history = []) {
  const body = { question };
  if (companies) body.companies = companies;
  if (history.length) body.history = history;   // earlier questions of this chat, for follow-ups
  let res;
  try {
    res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Could not reach the server. Check that it is running, then try again.");
  }
  if (!res.ok || !res.body) throw new ApiError(`The server answered ${res.status}.`, res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {   // one SSE event ends with a blank line
      const { event, data } = parseEvent(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      onEvent(event, data);
    }
  }
}

function parseEvent(raw) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data: data ? JSON.parse(data) : {} };
}
