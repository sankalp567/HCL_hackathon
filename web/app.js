// Ask a question, read the server-sent events from POST /api/ask, and show the answer with its citations.
const $ = (id) => document.getElementById(id);
const form = $("ask-form"), input = $("question"), button = $("ask-button");
let sources = {};     // n -> source, from the `meta` event
let citations = {};   // n -> citation, from the `citations` event
let answerText = "";

form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value.trim()); });
document.querySelectorAll(".examples .chip").forEach((chip) =>
  chip.addEventListener("click", () => { input.value = chip.textContent; ask(chip.textContent); }));
$("panel-close").addEventListener("click", closePanel);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanel(); });

async function ask(question) {
  if (!question) return;
  reset();
  button.disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) throw new Error(`server answered ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {   // one SSE event ends with a blank line
        handle(parseEvent(buffer.slice(0, end)));
        buffer = buffer.slice(end + 2);
      }
    }
  } catch (err) {
    showCard("nf", "Something went wrong", String(err.message || err));
  } finally {
    button.disabled = false;
    $("answer").classList.remove("streaming");
  }
}

function parseEvent(raw) {
  let event = "message", data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data: data ? JSON.parse(data) : {} };
}

function handle({ event, data }) {
  if (event === "meta") {
    sources = Object.fromEntries(data.sources.map((s) => [s.n, s]));
    $("status").textContent = `Searching the reports… ${data.sources.length} passages and facts found`;
  } else if (event === "token") {
    answerText += data.t;
    $("status").textContent = "Answering from the reports";
    $("answer").classList.add("streaming");
    renderAnswer();
  } else if (event === "citations") {
    citations = Object.fromEntries(data.citations.map((c) => [c.n, c]));
    renderCitations(data.citations);
  } else if (event === "notice") {
    $("notice").textContent = data.message;
    $("notice").hidden = false;
  } else if (event === "error") {
    $("notice").textContent = data.message;
    $("notice").hidden = false;
  } else if (event === "done") {
    finish(data);
  }
}

function finish(done) {
  $("answer").classList.remove("streaming");
  $("footer").textContent = `${done.backend} · ${(done.latency_ms / 1000).toFixed(1)} s`;
  if (done.status === "answered") {
    $("status").textContent = "Answer from the reports";
  } else if (done.status === "out_of_scope") {
    showCard("oos", "Out of scope", done.message);
  } else if (done.status === "not_found") {
    showCard("nf", "Not found in the reports", done.message.replace(/^Not found in the reports[.:]?\s*/, ""));
  } else {
    showCard("nf", "Something went wrong", done.message);
  }
}

// The answer is built from text nodes and buttons only, never from HTML strings, so report text cannot inject markup.
function renderAnswer() {
  const box = $("answer");
  box.replaceChildren();
  answerText.split(/(\[\d+\])/).forEach((part) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return box.append(document.createTextNode(part));
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "marker";
    chip.textContent = m[1];
    chip.setAttribute("aria-label", `Source ${m[1]}`);
    chip.addEventListener("click", () => openPanel(citations[m[1]] || sources[m[1]]));
    box.append(chip);
  });
}

function renderCitations(list) {
  const ol = $("citations");
  ol.replaceChildren();
  for (const c of list) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = '<span class="n"></span><span class="where"></span><span class="tick"></span>';
    b.querySelector(".n").textContent = c.n;
    b.querySelector(".where").textContent = `${c.company} · ${c.report} · PDF p. ${c.page}`;
    const tick = b.querySelector(".tick");
    tick.textContent = c.verified ? "✓ numbers match" : "check the page";
    if (!c.verified) tick.classList.add("no");
    b.addEventListener("click", () => openPanel(c));
    li.append(b);
    ol.append(li);
  }
}

function showCard(kind, title, message) {
  answerText = "";
  const card = document.createElement("div");
  card.className = `card ${kind}`;
  const strong = document.createElement("strong");
  strong.textContent = title;
  card.append(strong, document.createTextNode(message || ""));
  $("answer").replaceChildren(card);
  $("citations").replaceChildren();
  $("status").textContent = "";
}

function openPanel(c) {
  if (!c) return;
  const bbox = c.bbox ? `?x0=${c.bbox[0]}&y0=${c.bbox[1]}&x1=${c.bbox[2]}&y1=${c.bbox[3]}` : "";
  $("panel-title").textContent = `${c.company} · ${c.report} · PDF p. ${c.page}`;
  $("panel-quote").textContent = c.quote || "";
  $("panel-link").href = `/reports/${encodeURIComponent(c.pdf)}#page=${c.page}`;
  $("panel-image").src = `/api/page/${encodeURIComponent(c.doc_id)}/${c.page}.png${bbox}`;
  $("panel").hidden = false;
  $("panel-close").focus();
}

function closePanel() { $("panel").hidden = true; }

function reset() {
  sources = {}; citations = {}; answerText = "";
  $("result").hidden = false;
  $("answer").replaceChildren();
  $("citations").replaceChildren();
  $("notice").hidden = true;
  $("footer").textContent = "";
  $("status").textContent = "Searching the reports…";
}
