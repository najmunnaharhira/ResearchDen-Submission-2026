const input = document.getElementById("input");
const btn = document.getElementById("check-btn");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const wordsEl = document.getElementById("words");
const styloEl = document.getElementById("stylo");
const noteEl = document.getElementById("note");
const examplesList = document.getElementById("examples-list");

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const AI_RGB = hexToRgb("#C2410C");
const HUMAN_RGB = hexToRgb("#0F766E");

function renderLabel(pAi, pHuman) {
  labelEl.innerHTML = "";
  const rows = [
    { name: "AI-generated", pct: pAi, color: "#C2410C" },
    { name: "Human-written", pct: pHuman, color: "#0F766E" },
  ];
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "label-row";
    row.innerHTML = `
      <span class="name">${r.name}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(r.pct * 100).toFixed(1)}%;background:${r.color}"></span></span>
      <span class="pct">${(r.pct * 100).toFixed(1)}%</span>
    `;
    labelEl.appendChild(row);
  }
}

function renderWords(segments) {
  wordsEl.innerHTML = "";
  for (const seg of segments) {
    if (seg.score === null || seg.score === undefined) {
      wordsEl.appendChild(document.createTextNode(seg.text));
      continue;
    }
    const span = document.createElement("span");
    span.className = "scored";
    const rgb = seg.score >= 0 ? AI_RGB : HUMAN_RGB;
    const alpha = Math.min(Math.abs(seg.score), 1) * 0.55;
    span.style.backgroundColor = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
    span.title = `score: ${seg.score}`;
    span.textContent = seg.text;
    wordsEl.appendChild(span);
  }
}

function renderStylo(stylo) {
  styloEl.innerHTML = "";
  const top = stylo.slice(0, 8);
  const maxAbs = Math.max(...top.map((s) => Math.abs(s.score)), 1e-9);
  for (const s of top) {
    const pct = (Math.abs(s.score) / maxAbs) * 50;
    const rgb = s.score >= 0 ? AI_RGB : HUMAN_RGB;
    const row = document.createElement("div");
    row.className = "stylo-row";
    const side = s.score >= 0 ? `left:50%;width:${pct}%;` : `right:50%;width:${pct}%;`;
    row.innerHTML = `
      <span class="label">${s.label}</span>
      <span class="stylo-bar-wrap">
        <span class="stylo-bar" style="${side}background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>
      </span>
    `;
    styloEl.appendChild(row);
  }
}

async function analyze() {
  const text = input.value.trim();
  if (!text) {
    statusEl.textContent = "Paste some Banglish text first.";
    return;
  }
  btn.disabled = true;
  statusEl.textContent = "Checking...";
  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Something went wrong.");
    renderLabel(data.p_ai, data.p_human);
    renderWords(data.segments);
    renderStylo(data.stylo);
    noteEl.textContent = data.note;
    noteEl.className = "note" + (data.short_text ? " warn" : "");
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", analyze);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

async function loadExamples() {
  try {
    const resp = await fetch("/api/examples");
    const data = await resp.json();
    examplesList.innerHTML = "";
    for (const ex of data.examples) {
      const div = document.createElement("div");
      div.className = "example-item";
      div.textContent = ex;
      div.addEventListener("click", () => {
        input.value = ex;
        analyze();
      });
      examplesList.appendChild(div);
    }
  } catch (_) {
    /* examples are a nice-to-have, ignore failures */
  }
}
loadExamples();
