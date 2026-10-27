// Ask AI: the chat thread, the streamed answer with [n] citation chips, the status cards,
// and the chart card for questions about a metric.
import { api, askStream } from "./api.js";
import { metricPanel } from "./charts.js";
import { citationRef, factRef, showEvidence, sourceRef } from "./evidence.js";
import { allCompaniesSelected, state } from "./state.js";
import { h, icon, metricName, prefersReducedMotion, spinner, swatch } from "./util.js";

export const SUGGESTIONS = [
  "Compare revenue of TCS, Infosys and HCLTech",
  "Compare operating margins",
  "Show net profit comparison",
  "Compare headcount",
  "What was TCS's revenue in FY2029-30?",
  "Should I buy Infosys shares?",
];

// done.backend -> how the answer was produced, in words.
const BACKENDS = {
  facts: "Answered from extracted data",
  vllm: "Answered by the self-hosted model (vLLM)",
  transformers: "Answered by the self-hosted model",
  extractive: "Extractive answer, no model",
};

const PANEL_EMPTY = {
  not_found: "No sources: nothing in the reports supports an answer to this question.",
  out_of_scope: "No sources: this question is outside what the annual reports can answer.",
  error: "No sources: the answer did not complete.",
};

const thread = document.getElementById("thread");
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const button = document.getElementById("ask-button");
let busy = false;

export function initChat() {
  document.getElementById("suggestions").replaceChildren(
    ...SUGGESTIONS.map((q) => h("button", { type: "button", class: "chip", onclick: () => ask(q) }, q)));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(input.value);
  });
  updateScopeHint();
}

/** The empty thread lists the three reports the answers come from. */
export function renderWelcome(reports) {
  const welcome = document.getElementById("welcome");
  if (!welcome || !reports?.length) return;
  welcome.append(h("ul", { class: "welcome-reports", "aria-label": "Reports loaded" }, reports.map((r) => h("li", null,
    swatch(r.company),
    h("div", null, h("strong", null, r.company), h("span", { class: "muted" }, r.title), h("span", { class: "muted" }, `${r.pages} pages`))))));
}

/** The company filter only scopes questions that name no company; say so when it is active. */
export function updateScopeHint() {
  const hint = document.getElementById("scope-hint");
  hint.hidden = allCompaniesSelected();
  hint.textContent = `Questions that name no company cover ${state.companies.join(" and ")} (set in the top bar).`;
}

export async function ask(rawQuestion) {
  const question = rawQuestion.trim();
  if (!question || busy) return;
  busy = true;
  button.disabled = true;
  input.value = "";
  document.getElementById("welcome")?.remove();
  thread.append(h("article", { class: "msg msg-user", "aria-label": "Your question" },
    h("div", { class: "bubble bubble-user" }, question),
    h("div", { class: "avatar avatar-user", "aria-hidden": "true" }, icon("user"))));
  const answer = new Answer(question);
  thread.append(answer.el);
  scrollDown(true);
  try {
    await askStream(question, allCompaniesSelected() ? null : state.companies, (event, data) => {
      answer.handle(event, data);
      scrollDown();
    });
    if (!answer.finished) answer.fail("The connection closed before the answer finished.");
  } catch (err) {
    answer.fail(err.message);
  } finally {
    busy = false;
    button.disabled = false;
  }
}

function scrollDown(force = false) {
  const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 140;
  if (force || nearBottom) thread.scrollTop = thread.scrollHeight;
}

/** One assistant message, filled in as the server-sent events arrive. */
class Answer {
  constructor(question) {
    this.question = question;
    this.text = "";
    this.sentences = [];      // from `final`: [{text, status}]
    this.sources = {};        // n -> retrieved source, from `meta`
    this.citations = [];      // from `citations`
    this.items = [];          // what the evidence panel shows for this answer
    this.understood = null;
    this.finished = false;
    this.errorDetail = "";

    this.status = h("p", { class: "msg-status", role: "status" }, spinner(), h("span", null, "Searching the reports…"));
    this.notice = h("div", { class: "notice", hidden: true });
    this.textBox = h("div", { class: "answer-text", "aria-live": "polite", "aria-busy": "true" });
    this.bubble = h("div", { class: "bubble bubble-ai", hidden: true }, this.textBox);
    this.removed = h("div", { class: "removed-wrap" });
    this.extra = h("div", { class: "msg-extra" });
    this.meta = h("p", { class: "msg-meta" });
    this.el = h("article", { class: "msg msg-ai", "aria-label": "Answer" },
      h("div", { class: "avatar avatar-ai", "aria-hidden": "true" }, icon("bot")),
      h("div", { class: "msg-body" }, this.status, this.notice, this.bubble, this.removed, this.extra, this.meta));
  }

  handle(event, data) {
    if (event === "meta") {
      this.understood = data.understood;
      this.sources = Object.fromEntries(data.sources.map((s) => [s.n, s]));
      if (data.sources.length) this.setStatus(`Found ${data.sources.length} sources in the reports. Writing the answer…`);
    } else if (event === "token") {
      this.text += data.t;
      this.bubble.hidden = false;
      this.textBox.classList.add("streaming");
      this.renderText();
    } else if (event === "final") {
      // The checked answer replaces the streamed text: citations may be re-pointed, unsupported sentences dropped.
      this.text = data.answer || "";
      this.sentences = data.sentences || [];
      this.bubble.hidden = !this.text;
      this.renderText();
      this.renderRemoved();
    } else if (event === "citations") {
      this.citations = data.citations;
      this.items = data.citations.map(citationRef);
      showEvidence(this.items, { view: "ask" });
      this.renderText();   // chips can now name their source
    } else if (event === "notice") {
      this.notice.hidden = false;
      this.notice.replaceChildren(icon("info"), h("span", null, data.message));
    } else if (event === "error") {
      this.errorDetail = data.message || "";
    } else if (event === "done") {
      this.finish(data);
    }
  }

  setStatus(text) {
    this.status.lastChild.textContent = text;
  }

  renderText() {
    const text = this.text.replace(/[ \t]+\n/g, "\n").trim();
    const uncited = this.sentences.filter((s) => s.status === "uncited").map((s) => s.text);
    const nodes = [];
    let pos = 0;
    for (const sentence of uncited) {   // mark sentences that cite no source, in place
      const at = text.indexOf(sentence, pos);
      if (at < 0) continue;
      nodes.push(...this.inline(text.slice(pos, at)),
        h("span", { class: "uncited" }, this.inline(sentence), " ", h("span", { class: "tag", title: "This sentence cites no source in the reports" }, "no source")));
      pos = at + sentence.length;
    }
    nodes.push(...this.inline(text.slice(pos)));
    this.textBox.replaceChildren(...nodes);
  }

  /**
   * Text with [n] markers turned into citation chips, and **bold** kept bold. Never parsed as HTML.
   * Each chip is kept on one line with the word before it and the punctuation after it.
   */
  inline(text) {
    const parts = text.split(/(\[\d+\])/);
    const nodes = [];
    for (let i = 0; i < parts.length; i++) {
      const marker = parts[i].match(/^\[(\d+)\]$/);
      if (!marker) {
        nodes.push(...bold(parts[i]));
        continue;
      }
      const before = typeof nodes.at(-1)?.data === "string" ? nodes.pop().data : "";
      const [, lead, word] = before.match(/^([\s\S]*?)(\S*[ \t]*)$/);   // the last word and the space after it
      const [punct = ""] = (parts[i + 1] || "").match(/^[.,;:)]+/) || [];
      if (punct) parts[i + 1] = parts[i + 1].slice(punct.length);
      if (lead) nodes.push(document.createTextNode(lead));
      nodes.push(h("span", { class: "nowrap" }, word, this.chip(Number(marker[1])), punct));
    }
    return nodes;
  }

  chip(n) {
    const source = this.citations.find((c) => c.n === n) || this.sources[n];
    const where = source ? `${source.company}, page ${source.page}` : "";
    return h("button", {
      type: "button", class: "cite", title: where || null, "aria-label": `Source ${n}${where ? `: ${where}` : ""}`,
      onclick: () => this.openSource(n),
    }, String(n));
  }

  openSource(n) {
    const item = this.items.find((i) => i.n === n);
    if (item) {
      showEvidence(this.items, { view: "ask", focus: item.key, open: true });
    } else if (this.sources[n]) {   // citations not in yet: show the retrieved source
      const ref = sourceRef(this.sources[n]);
      showEvidence([ref], { view: "ask", focus: ref.key, open: true });
    }
  }

  renderRemoved() {
    const removed = this.sentences.filter((s) => s.status === "removed");
    this.removed.replaceChildren(removed.length ? h("details", { class: "removed" },
      h("summary", null, `Removed: not supported by the reports (${removed.length})`),
      h("ul", null, removed.map((s) => h("li", null, s.text.replace(/\s*\[\d+\]/g, ""))))) : "");
  }

  finish(done) {
    this.finished = true;
    this.status.remove();
    this.textBox.classList.remove("streaming");
    this.textBox.setAttribute("aria-busy", "false");
    const took = done.latency_ms < 1000 ? `${done.latency_ms} ms` : `${(done.latency_ms / 1000).toFixed(1)} s`;

    if (done.status === "answered") {
      const cited = this.citations.length;
      const verified = this.citations.filter((c) => c.verified).length;
      this.meta.replaceChildren(icon("check"),
        `${cited} source${cited === 1 ? "" : "s"} cited, ${verified} with matching numbers · ${BACKENDS[done.backend] || done.backend} · ${took}`);
      if (this.understood?.metric) this.addMetricCard();
      return;
    }

    // Anything else is not an answer, so no answer text stays on screen.
    this.bubble.hidden = true;
    this.removed.replaceChildren();
    this.items = [];
    showEvidence([], { view: "ask", empty: PANEL_EMPTY[done.status] || PANEL_EMPTY.error });
    this.meta.replaceChildren(`Checked in ${took}`);
    if (done.status === "not_found") {
      const reason = (done.message || "").replace(/^Not found in the reports[.:]?\s*/i, "");
      const card = resultCard("not-found", "alert", "Not found in the reports",
        reason ? reason[0].toUpperCase() + reason.slice(1) : "The reports do not answer this question.");
      this.extra.append(card);
      this.addAvailability(card);
    } else if (done.status === "out_of_scope") {
      this.extra.append(resultCard("out-of-scope", "ban", "Out of scope",
        done.message || "This question is outside what the annual reports can answer.",
        "Ask about what the reports contain, such as revenue, margins, headcount or attrition."));
    } else {
      this.extra.append(resultCard("error", "alert", "Something went wrong",
        [done.message, this.errorDetail].filter(Boolean).join(" "), null, this.retryButton()));
    }
  }

  /** For a metric asked in a year the reports lack, say which years the extracted data does have. */
  async addAvailability(card) {
    const { metric, fy, companies } = this.understood || {};
    if (!metric || !fy) return;
    try {
      const data = await api.metrics(metric);
      const parts = companies.map((c) => {
        const years = [...new Set(data.facts.filter((f) => f.company === c).map((f) => f.fy))].sort();
        return years.length ? `${c} ${years.join(", ")}` : null;
      }).filter(Boolean);
      if (parts.length) {
        card.lastChild.append(h("p", { class: "result-hint" }, `Years with ${metricName(metric).toLowerCase()} in the extracted data: ${parts.join("; ")}.`));
      }
    } catch { /* the hint is optional */ }
  }

  fail(message) {
    if (this.finished) return;
    this.finished = true;
    this.status.remove();
    this.textBox.classList.remove("streaming");
    this.textBox.setAttribute("aria-busy", "false");
    this.bubble.hidden = true;
    this.extra.append(resultCard("error", "alert", "The answer did not complete", message, null, this.retryButton()));
  }

  retryButton() {
    return h("button", { type: "button", class: "btn btn-ghost btn-sm", onclick: () => ask(this.question) }, "Ask again");
  }

  addMetricCard() {
    const { metric, companies, fy } = this.understood;
    const panel = metricPanel({
      metric, companies, year: "all", layout: "tabs", highlightFy: fy,
      onPick: (facts, m) => {
        // Chart and table numbers join this answer's evidence list, next to the cited sources.
        const refs = facts.map((f) => factRef(f, m));
        const known = new Set(this.items.map((i) => i.key));
        this.items = [...this.items, ...refs.filter((r) => !known.has(r.key))];
        showEvidence(this.items, { view: "ask", focus: refs[0].key, open: true });
      },
    });
    this.extra.append(panel.root);
    // Bring the answer's first line into view, with the chart card below it.
    requestAnimationFrame(() => this.el.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" }));
  }
}

/** **bold** pieces as <strong>, the rest as text nodes. */
function bold(text) {
  return text.split(/\*\*(.+?)\*\*/).map((piece, i) => (i % 2 ? h("strong", null, piece) : document.createTextNode(piece)));
}

function resultCard(kind, iconName, title, text, hint = null, action = null) {
  return h("div", { class: `result-card ${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(iconName),
    h("div", null,
      h("p", { class: "result-title" }, title),
      text ? h("p", { class: "result-text" }, text) : "",
      hint ? h("p", { class: "result-hint" }, hint) : "",
      action || ""));
}
