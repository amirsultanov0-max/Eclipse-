"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en-US");

// One-hue sequential ramp (blue, steps 100 -> 700). In dark mode it runs the
// other way, so near-zero weights recede toward the surface in both themes.
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const ramp = () => (darkQuery.matches ? [...RAMP].reverse() : RAMP);

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function weightColor(w) {
  const steps = ramp();
  const pos = Math.min(Math.max(w, 0), 1) * (steps.length - 1);
  const i = Math.min(Math.floor(pos), steps.length - 2);
  const t = pos - i;
  const a = hexToRgb(steps[i]);
  const b = hexToRgb(steps[i + 1]);
  const mix = a.map((v, k) => Math.round(v + (b[k] - v) * t));
  return `rgb(${mix.join(",")})`;
}

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, content, attrs = {}) {
  const node = el("text", { x, y, ...attrs }, parent);
  node.textContent = content;
  return node;
}

// --- architecture diagram ------------------------------------------------------

function node(svg, x, y, w, h, title, detail, extra = "") {
  const g = el("g", { class: `node ${extra}` }, svg);
  el("rect", { x, y, width: w, height: h, rx: 8 }, g);
  text(g, x + w / 2, y + (detail ? h / 2 - 4 : h / 2 + 5), title,
       { class: "title", "text-anchor": "middle" });
  if (detail) text(g, x + w / 2, y + h / 2 + 14, detail, { class: "detail", "text-anchor": "middle" });
  return g;
}

function arrow(svg, x1, y1, x2, y2, cls = "edge") {
  el("path", { d: `M${x1},${y1} L${x2},${y2}`, class: cls, "marker-end": "url(#arrowhead)" }, svg);
}

function plus(svg, cx, cy) {
  const g = el("g", { class: "plus" }, svg);
  el("circle", { cx, cy, r: 13 }, g);
  text(g, cx, cy + 5, "+", { "text-anchor": "middle" });
}

function renderDiagram(a) {
  const arch = a.architecture;
  const c = a.components;
  const block = c.blocks[0];
  const perBlockSame = c.blocks.every((b) => b.total === block.total);
  const blocksTotal = c.blocks.reduce((s, b) => s + b.total, 0);

  const svg = el("svg", { viewBox: "0 0 640 880", xmlns: SVG_NS });
  const defs = el("defs", {}, svg);
  const marker = el("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 9, refY: 5,
                                markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z" }, marker);

  node(svg, 170, 10, 300, 48, "Prompt tokens",
       `up to ${fmt(arch.context_length)} tokens from a ${fmt(arch.vocab_size)}-token vocabulary`);
  arrow(svg, 290, 58, 170, 88);
  arrow(svg, 350, 58, 470, 88);
  node(svg, 40, 90, 260, 56, "Token embedding",
       `${fmt(arch.vocab_size)} × ${arch.d_model} · ${fmt(c.token_embedding)} params`);
  node(svg, 340, 90, 260, 56, "Position embedding",
       `${fmt(arch.context_length)} × ${arch.d_model} · ${fmt(c.position_embedding)} params`);
  arrow(svg, 170, 146, 309, 172);
  arrow(svg, 470, 146, 331, 172);
  plus(svg, 320, 182);
  arrow(svg, 320, 195, 320, 213);

  el("rect", { x: 60, y: 215, width: 520, height: 385, rx: 12, class: "container" }, svg);
  text(svg, 76, 238, `Transformer block × ${arch.num_blocks}`, { class: "container-label" });
  text(svg, 564, 238, perBlockSame
    ? `${fmt(block.total)} params each · ${fmt(blocksTotal)} in all ${arch.num_blocks}`
    : `${fmt(blocksTotal)} params in all ${arch.num_blocks}`,
       { class: "container-detail", "text-anchor": "end" });

  arrow(svg, 320, 248, 320, 260);
  node(svg, 180, 262, 280, 40, "LayerNorm", `${fmt(block.ln_1)} params`);
  arrow(svg, 320, 302, 320, 320);
  const link = el("a", { href: "#attention_section" }, svg);
  node(link, 180, 322, 280, 56, "Causal self-attention",
       `${arch.num_heads} heads × ${arch.head_dim} dims · ${fmt(block.attention)} params`, "accent");
  text(svg, 470, 354, "see section 2 ↓", { class: "side-note accent" });
  arrow(svg, 320, 378, 320, 387);
  plus(svg, 320, 402);
  el("path", { d: "M320,252 H130 V402 H305", class: "residual", "marker-end": "url(#arrowhead)" }, svg);

  arrow(svg, 320, 415, 320, 430);
  node(svg, 180, 432, 280, 40, "LayerNorm", `${fmt(block.ln_2)} params`);
  arrow(svg, 320, 472, 320, 490);
  node(svg, 180, 492, 280, 56, "Feed-forward",
       `${arch.d_model} → ${fmt(arch.d_ff)} → ${arch.d_model}, GELU · ${fmt(block.feed_forward)} params`);
  arrow(svg, 320, 548, 320, 557);
  plus(svg, 320, 572);
  el("path", { d: "M320,422 H130 V572 H305", class: "residual", "marker-end": "url(#arrowhead)" }, svg);
  text(svg, 122, 490, "residual (skip) connections", {
    class: "side-note", "text-anchor": "middle", transform: "rotate(-90 122 490)" });

  arrow(svg, 320, 585, 320, 628);
  node(svg, 180, 630, 280, 40, "Final LayerNorm", `${fmt(c.ln_final)} params`);
  arrow(svg, 320, 670, 320, 698);
  node(svg, 180, 700, 280, 56, "Output head",
       `reuses the token embedding · ${fmt(c.output_head)} new params`);
  el("path", { d: "M180,728 H22 V118 H38", class: "tied", "marker-end": "url(#arrowhead)" }, svg);
  text(svg, 14, 420, "same matrix (tied weights)", {
    class: "side-note accent", "text-anchor": "middle", transform: "rotate(-90 14 420)" });
  arrow(svg, 320, 756, 320, 784);
  node(svg, 170, 786, 300, 48, "Next-token probabilities",
       `softmax over ${fmt(arch.vocab_size)} tokens`);
  text(svg, 320, 866, `Total: ${fmt(a.parameters)} parameters`,
       { class: "container-label", "text-anchor": "middle" });

  const holder = $("diagram");
  holder.replaceChildren(svg);
  holder.setAttribute("aria-label",
    `Architecture diagram: token and position embeddings (${arch.d_model} dimensions), ` +
    `${arch.num_blocks} Transformer blocks each with ${arch.num_heads}-head causal self-attention ` +
    `and a ${fmt(arch.d_ff)}-wide feed-forward layer, a final LayerNorm, and an output head tied ` +
    `to the token embedding. ${fmt(a.parameters)} parameters in total.`);
  $("arch_summary").textContent =
    `Loaded checkpoint: ${a.checkpoint}, step ${a.step == null ? "unknown" : fmt(a.step)}. ` +
    `d_model ${arch.d_model}, ${arch.num_heads} heads of ${arch.head_dim}, d_ff ${fmt(arch.d_ff)}, ` +
    `${arch.num_blocks} blocks, ${fmt(arch.context_length)}-token context, ` +
    `${fmt(arch.vocab_size)}-token vocabulary, ${fmt(a.parameters)} parameters.`;
}

async function loadArchitecture() {
  try {
    const response = await fetch("/api/architecture");
    if (!response.ok) throw new Error(`status ${response.status}`);
    renderDiagram(await response.json());
  } catch (err) {
    $("diagram").textContent = "Could not load the architecture from the server.";
  }
}

// --- attention -------------------------------------------------------------------

const state = { data: null, block: 0, head: 0 };
const tooltip = $("tooltip");

function showToken(tok) {
  return tok.replace(/\n/g, "↵").replace(/ /g, "·");
}

function shortToken(tok) {
  const shown = showToken(tok);
  return shown.length > 12 ? shown.slice(0, 11) + "…" : shown;
}

function renderHeatmap() {
  const { data, block, head } = state;
  const tokens = data.tokens;
  const T = tokens.length;
  const weights = data.attention[block][head];
  const cell = 22;
  const labelChars = Math.max(...tokens.map((t) => shortToken(t).length));
  const left = Math.min(110, labelChars * 7 + 10);
  const top = Math.min(90, labelChars * 6 + 14);
  const width = left + T * cell + 4;
  const height = top + T * cell + 4;

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, xmlns: SVG_NS,
                          role: "img",
                          "aria-label": `Attention weights, block ${block + 1}, head ${head + 1}` });
  svg.style.maxWidth = `${Math.round(width * 1.6)}px`;
  const rowLabels = [];
  const colLabels = [];
  tokens.forEach((tok, i) => {
    rowLabels.push(text(svg, left - 6, top + i * cell + cell / 2 + 4, shortToken(tok),
                        { "text-anchor": "end" }));
    const x = left + i * cell + cell / 2 + 3;
    colLabels.push(text(svg, x, top - 6, shortToken(tok),
                        { transform: `rotate(-55 ${x} ${top - 6})` }));
  });

  for (let r = 0; r < T; r++) {
    for (let c = 0; c <= r; c++) {
      el("rect", {
        class: "cell", x: left + c * cell + 1, y: top + r * cell + 1,
        width: cell - 2, height: cell - 2, rx: 2,
        fill: weightColor(weights[r][c]), "data-r": r, "data-c": c,
      }, svg);
    }
  }

  let active = null;
  svg.addEventListener("pointermove", (event) => {
    const target = event.target;
    if (!target.classList || !target.classList.contains("cell")) {
      hideTooltip();
      return;
    }
    const r = Number(target.dataset.r);
    const c = Number(target.dataset.c);
    if (active) active.forEach((n) => n.classList.remove("active"));
    active = [rowLabels[r], colLabels[c]];
    active.forEach((n) => n.classList.add("active"));
    const w = weights[r][c];
    tooltip.textContent =
      `Block ${block + 1}, head ${head + 1}: "${showToken(tokens[r])}" gives ` +
      `"${showToken(tokens[c])}" weight ${w.toFixed(4)} (${(w * 100).toFixed(1)}% of its attention)`;
    tooltip.hidden = false;
    const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${event.clientY + 16}px`;
  });
  svg.addEventListener("pointerleave", () => {
    if (active) active.forEach((n) => n.classList.remove("active"));
    active = null;
    hideTooltip();
  });

  $("heatmap").replaceChildren(svg);
  $("heat_title").textContent = `Block ${block + 1} of ${data.num_blocks}, head ${head + 1} of ${data.num_heads}`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function renderThumbs() {
  const { data, block, head } = state;
  const T = data.tokens.length;
  const holder = $("thumbs");
  holder.replaceChildren();
  data.attention[block].forEach((weights, h) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(h === head));
    button.setAttribute("aria-label", `Show head ${h + 1}`);
    const svg = el("svg", { viewBox: `0 0 ${T * 4} ${T * 4}`, "aria-hidden": "true" }, button);
    for (let r = 0; r < T; r++) {
      for (let c = 0; c <= r; c++) {
        el("rect", { x: c * 4, y: r * 4, width: 3.4, height: 3.4, fill: weightColor(weights[r][c]) }, svg);
      }
    }
    button.append(`Head ${h + 1}`);
    button.addEventListener("click", () => select(block, h));
    holder.appendChild(button);
  });
}

function renderPicker(holderId, label, count, current, onPick) {
  const holder = $(holderId);
  holder.replaceChildren();
  const caption = document.createElement("span");
  caption.className = "picker-label";
  caption.textContent = label;
  holder.appendChild(caption);
  for (let i = 0; i < count; i++) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(i === current));
    button.textContent = String(i + 1);
    button.setAttribute("aria-label", `${label} ${i + 1}`);
    button.addEventListener("click", () => onPick(i));
    holder.appendChild(button);
  }
}

function renderTable() {
  const { data, block, head } = state;
  const weights = data.attention[block][head];
  const table = $("attn_table");
  table.replaceChildren();
  const caption = document.createElement("caption");
  caption.textContent = `Block ${block + 1}, head ${head + 1}. Rows look at columns.`;
  caption.className = "caption";
  table.appendChild(caption);
  const header = table.insertRow();
  header.appendChild(document.createElement("th"));
  data.tokens.forEach((tok) => {
    const th = document.createElement("th");
    th.textContent = showToken(tok);
    header.appendChild(th);
  });
  weights.forEach((row, r) => {
    const tr = table.insertRow();
    const th = document.createElement("th");
    th.textContent = showToken(data.tokens[r]);
    tr.appendChild(th);
    row.forEach((w, c) => {
      tr.insertCell().textContent = c <= r ? w.toFixed(3) : "";
    });
  });
}

function select(block, head) {
  state.block = block;
  state.head = head;
  renderPicker("block_picker", "Block", state.data.num_blocks, block, (b) => select(b, state.head));
  renderPicker("head_picker", "Head", state.data.num_heads, head, (h) => select(state.block, h));
  renderHeatmap();
  renderThumbs();
  renderTable();
}

function renderLegend() {
  $("legend_bar").style.background = `linear-gradient(to right, ${ramp().join(", ")})`;
}

async function showAttention(event) {
  if (event) event.preventDefault();
  const input = $("attn_prompt");
  const errorBox = $("attn_error");
  errorBox.hidden = true;
  if (!input.value.trim()) {
    errorBox.textContent = "Type a short phrase first.";
    errorBox.hidden = false;
    return;
  }
  const button = $("attn_submit");
  button.disabled = true;
  try {
    const response = await fetch("/api/attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: input.value }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      errorBox.textContent = data.error || `The server returned an error (${response.status}).`;
      errorBox.hidden = false;
      return;
    }
    state.data = data;
    $("verification").textContent =
      `Verified: ${data.verification}. ${data.tokens.length} tokens. ` +
      `A · marks a space that is part of a token. ` +
      `${data.rounding.charAt(0).toUpperCase()}${data.rounding.slice(1)}.`;
    $("attn_result").hidden = false;
    renderLegend();
    select(Math.min(state.block, data.num_blocks - 1), Math.min(state.head, data.num_heads - 1));
  } catch (err) {
    errorBox.textContent = "Could not reach the Eclipse server. Is it still running?";
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
}

$("attn_form").addEventListener("submit", showAttention);
for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    $("attn_prompt").value = button.textContent;
    showAttention();
  });
}
darkQuery.addEventListener("change", () => {
  if (state.data) {
    renderLegend();
    select(state.block, state.head);
  }
});

loadArchitecture();
showAttention();
