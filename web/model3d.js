import * as THREE from "three";
import { OrbitControls } from "/static/vendor/three-0.185.0/OrbitControls.js";

// Every position, line and number drawn here comes from the server:
//   /api/embedding-projection  PCA coordinates and cosine neighbours of the token embeddings
//   /api/embedding-neighbors   one token's neighbours across the whole vocabulary
//   /api/attention             verified attention weights for one forward pass
// Colours, glow, sizes and motion are presentation only.

const $ = (id) => document.getElementById(id);
const RADIUS = 10;
const NODE = new THREE.Color("#86b6ef");
const NODE_HOT = new THREE.Color("#ffffff");
const NODE_NEAR = new THREE.Color("#cde2fb");
const EDGE = new THREE.Color("#9ec5f4");
const EDGE_HOT = new THREE.Color("#8f9ef5");
const PARTICLE = new THREE.Color("#cde2fb");
const LEGEND = "linear-gradient(to right, #0b0c10, #9ec5f4)";

// Read-only snapshot of what is on screen, for automated checks.
const debug = { embed: null, attn: null };
window.__eclipse3d = debug;

const shown = (tok) => tok.replace(/\n/g, "↵").replace(/ /g, "·");
const fmtNum = (x) => x.toFixed(4);

function glowTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.25, "rgba(255,255,255,0.85)");
  g.addColorStop(0.55, "rgba(255,255,255,0.18)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeStage(stageId, cameraZ) {
  const stage = $(stageId);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(new THREE.Color("#0b0c10"), 1);
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
  camera.position.set(0, 0, cameraZ);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = 4;
  controls.maxDistance = 80;
  let idleTimer = null;
  const view = { stage, renderer, scene, camera, controls, labels: stage.querySelector(".labels"),
                 wantsRotate: false, interacting: false, active: false, onFrame: null };
  controls.addEventListener("start", () => {
    view.interacting = true;
    clearTimeout(idleTimer);
  });
  controls.addEventListener("end", () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { view.interacting = false; }, 3000);
  });
  const resize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Keep a 50° field of view across the narrower side, so portrait screens see the whole scene.
    const half = THREE.MathUtils.degToRad(25);
    camera.fov = camera.aspect >= 1 ? 50
      : THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(half) / camera.aspect));
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(stage);
  resize();
  return view;
}

function toScreen(view, point) {
  const v = point.clone().project(view.camera);
  if (v.z > 1) return null;
  return { x: (v.x + 1) / 2 * view.stage.clientWidth, y: (1 - v.y) / 2 * view.stage.clientHeight };
}

function placeLabels(view, items) {
  // items: [{ position: Vector3, parts: [{ text, num }], cls }]; text is set via textContent.
  view.labels.replaceChildren();
  for (const item of items) {
    const p = toScreen(view, item.position);
    if (!p) continue;
    const el = document.createElement("div");
    el.className = `label3d ${item.cls || ""}`;
    for (const part of item.parts) {
      const span = document.createElement("span");
      if (part.num) span.className = "num";
      span.textContent = part.text;
      el.appendChild(span);
    }
    el.style.left = `${p.x}px`;
    el.style.top = `${p.y}px`;
    view.labels.appendChild(el);
  }
  // Nudge labels that overlap an earlier one downwards, so none hides another.
  const placed = [];
  for (const el of view.labels.children) {
    let rect = el.getBoundingClientRect();
    for (let tries = 0; tries < 6 && placed.some((r) => overlaps(r, rect)); tries++) {
      el.style.top = `${parseFloat(el.style.top) + rect.height + 2}px`;
      rect = el.getBoundingClientRect();
    }
    placed.push(rect);
  }
}

function overlaps(a, b) {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

function pointsObject(positions, size) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const colors = new Float32Array(positions.length);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size, map: glowTexture(), vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}

function lineObject(positions, colors, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  return new THREE.LineSegments(geometry, material);
}

function pushSegmentPath(points, color, positions, colors) {
  for (let i = 0; i < points.length - 1; i++) {
    positions.push(...points[i].toArray(), ...points[i + 1].toArray());
    colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
  }
}

function hoverIndex(view, points, event, threshold) {
  const rect = view.renderer.domElement.getBoundingClientRect();
  const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
                                    -((event.clientY - rect.top) / rect.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = threshold;
  ray.setFromCamera(pointer, view.camera);
  const hits = ray.intersectObject(points);
  return hits.length ? hits[0].index : null;
}

// --- embedding sphere ----------------------------------------------------------------

const embed = { view: null, data: null, layout: "sphere", positions: [], points: null,
                lines: null, edges: [], hovered: null, neighborCache: new Map() };

function layoutPositions(data, layout) {
  const raw = data.tokens.map((t) => new THREE.Vector3(...t.xyz));
  if (layout === "sphere") return raw.map((v) => v.clone().normalize().multiplyScalar(RADIUS));
  // Raw PCA: one uniform scale for the whole cloud, so relative geometry is untouched.
  const norms = raw.map((v) => v.length()).sort((a, b) => a - b);
  const p95 = norms[Math.floor(norms.length * 0.95)] || 1;
  return raw.map((v) => v.clone().multiplyScalar(RADIUS / p95));
}

function arcPoints(a, b, layout) {
  if (layout !== "sphere") return [a, b];
  const ua = a.clone().normalize();
  const ub = b.clone().normalize();
  const angle = ua.angleTo(ub);
  if (angle < 1e-4) return [a, b];
  const steps = Math.max(2, Math.ceil(angle / 0.12));
  const out = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const s = Math.sin(angle);
    const p = ua.clone().multiplyScalar(Math.sin((1 - t) * angle) / s)
      .add(ub.clone().multiplyScalar(Math.sin(t * angle) / s));
    out.push(p.multiplyScalar(RADIUS * 1.002));
  }
  return out;
}

function disposeObject(obj) {
  if (!obj) return;
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    }
  });
  if (obj.parent) obj.parent.remove(obj);
}

function buildEmbedding() {
  const { view, data } = embed;
  for (const obj of [embed.points, embed.lines, embed.highlight]) disposeObject(obj);
  embed.positions = layoutPositions(data, embed.layout);
  embed.index = new Map(data.tokens.map((t, i) => [t.id, i]));

  embed.points = pointsObject(embed.positions.flatMap((v) => v.toArray()), 0.95);
  view.scene.add(embed.points);

  const seen = new Map();
  data.tokens.forEach((t, i) => {
    for (const [nid, sim] of t.neighbors) {
      const j = embed.index.get(nid);
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!seen.has(key)) seen.set(key, { a: Math.min(i, j), b: Math.max(i, j), sim });
    }
  });
  embed.edges = [...seen.values()];

  // The base sphere is drawn once per load and never changes on hover.
  const positions = [];
  const lineColors = [];
  for (const e of embed.edges) {
    const color = EDGE.clone().multiplyScalar(Math.max(0, Math.min(1, e.sim)));
    pushSegmentPath(arcPoints(embed.positions[e.a], embed.positions[e.b], embed.layout),
                    color, positions, lineColors);
  }
  embed.lines = lineObject(positions, lineColors, 0.4);
  view.scene.add(embed.lines);
  embed.highlight = null;
  colourEmbedding();
  debug.embed = {
    layout: embed.layout,
    tokens: data.tokens.map((t) => ({ id: t.id, text: t.text })),
    edges: embed.edges.map((e) => ({ a: data.tokens[e.a].id, b: data.tokens[e.b].id, sim: e.sim })),
  };
}

function ownNeighbors(i) {
  return embed.data.tokens[i].neighbors.map(([id, sim]) => ({ index: embed.index.get(id), sim }));
}

// Hover changes only the hovered point, its own k neighbours and the k lines to them.
function colourEmbedding() {
  const { data, hovered } = embed;
  const own = hovered === null ? [] : ownNeighbors(hovered);
  const ownSet = new Set(own.map((n) => n.index));
  const colors = embed.points.geometry.getAttribute("color");
  data.tokens.forEach((_, i) => {
    const c = i === hovered ? NODE_HOT : ownSet.has(i) ? NODE_NEAR : NODE;
    colors.setXYZ(i, c.r, c.g, c.b);
  });
  colors.needsUpdate = true;

  disposeObject(embed.highlight);
  embed.highlight = null;
  if (hovered === null) return;
  const positions = [];
  const lineColors = [];
  for (const n of own) {
    const color = NODE_NEAR.clone().multiplyScalar(0.35 + 0.65 * Math.max(0, Math.min(1, n.sim)));
    pushSegmentPath(arcPoints(embed.positions[hovered], embed.positions[n.index], embed.layout),
                    color, positions, lineColors);
  }
  embed.highlight = lineObject(positions, lineColors, 1);
  embed.view.scene.add(embed.highlight);
}

async function showEmbeddingInfo(i) {
  const { data } = embed;
  const token = data.tokens[i];
  const info = $("embed_info");
  info.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = `"${shown(token.text)}"`;
  const meta = document.createElement("div");
  meta.textContent = `token id ${token.id} · PCA (${token.xyz.map((v) => v.toFixed(3)).join(", ")})`;
  info.append(h, meta);
  const addTable = (title, rows) => {
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = title;
    const table = document.createElement("table");
    for (const [label, value] of rows) {
      const tr = table.insertRow();
      tr.insertCell().textContent = shown(label);
      const v = tr.insertCell();
      v.className = "v";
      v.textContent = fmtNum(value);
    }
    info.append(sub, table);
  };
  const byId = new Map(data.tokens.map((t) => [t.id, t.text]));
  addTable("Lines drawn (cosine)", token.neighbors.map(([id, sim]) => [byId.get(id), sim]));

  let full = embed.neighborCache.get(token.id);
  if (!full) {
    const response = await fetch(`/api/embedding-neighbors?token_id=${token.id}&k=8`);
    if (!response.ok) return;
    full = await response.json();
    embed.neighborCache.set(token.id, full);
  }
  if (embed.hovered === i) {
    addTable(`Nearest in ${full.scope}`, full.neighbors.map((n) => [n.text, n.similarity]));
  }
}

function embedLabels() {
  const { view, data, hovered } = embed;
  if (hovered === null) {
    view.labels.replaceChildren();
    return;
  }
  const items = [{ position: embed.positions[hovered], cls: "main",
                   parts: [{ text: shown(data.tokens[hovered].text) }] }];
  for (const n of ownNeighbors(hovered)) {
    items.push({ position: embed.positions[n.index], cls: "dim",
                 parts: [{ text: `${shown(data.tokens[n.index].text)} ` }, { text: fmtNum(n.sim), num: true }] });
  }
  placeLabels(view, items);
}

async function loadEmbedding() {
  const narrow = window.innerWidth < 600;
  const countSelect = $("count");
  if (narrow && !embed.data) countSelect.value = "100";
  const response = await fetch(`/api/embedding-projection?count=${countSelect.value}&k=${$("k").value}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `status ${response.status}`);
  embed.data = data;
  embed.hovered = null;
  $("embed_info").textContent = "Hover over a point to see its token and its nearest neighbours.";
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  $("variance").textContent =
    `These three directions capture ${pct(data.explained_variance_total)} of the variance in the ` +
    `${data.vocab_size.toLocaleString()} × ${data.d_model} embedding matrix ` +
    `(${data.explained_variance_ratio.map(pct).join(" + ")}), so this is a partial view.`;
  $("k_label").textContent = String(data.k);
  $("dmodel_label").textContent = String(data.d_model);
  $("selection_note").textContent = `Tokens shown: ${data.selection}.`;
  buildEmbedding();
  updateLayoutNote();
}

function updateLayoutNote() {
  $("layout_note").textContent = embed.layout === "sphere"
    ? "Unit sphere: every token is pushed out to the same radius along the direction of its three " +
      "PCA coordinates, so direction is real but distance from the centre is not."
    : "Raw PCA: tokens sit at their actual PCA coordinates, uniformly scaled to fit. " +
      "Lines are straight here; on the sphere they follow its surface.";
}

function initEmbedding() {
  embed.view = makeStage("stage_embed", 30);
  const { view } = embed;
  view.wantsRotate = $("rotate_embed").checked;
  view.active = true;
  view.onFrame = embedLabels;
  const canvas = view.renderer.domElement;
  const onPointer = (event) => {
    if (!embed.points) return;
    const i = hoverIndex(view, embed.points, event, 0.35);
    if (i === embed.hovered) return;
    embed.hovered = i;
    colourEmbedding();
    if (i === null) $("embed_info").textContent = "Hover over a point to see its token and its nearest neighbours.";
    else showEmbeddingInfo(i);
  };
  canvas.addEventListener("pointermove", onPointer);
  canvas.addEventListener("pointerdown", onPointer);
  $("rotate_embed").addEventListener("change", (e) => { view.wantsRotate = e.target.checked; });
  for (const button of document.querySelectorAll("[data-layout]")) {
    button.addEventListener("click", () => {
      embed.layout = button.dataset.layout;
      document.querySelectorAll("[data-layout]").forEach((b) =>
        b.setAttribute("aria-checked", String(b === button)));
      embed.hovered = null;
      buildEmbedding();
      updateLayoutNote();
    });
  }
  const reload = () => loadEmbedding().catch(showEmbedError);
  $("count").addEventListener("change", reload);
  $("k").addEventListener("change", reload);
  $("embed_legend").style.background = LEGEND;
  reload();
}

function showEmbedError(err) {
  $("variance").textContent = `Could not load the embeddings: ${err.message}`;
}

// --- attention sphere ------------------------------------------------------------------

const attn = { view: null, data: null, block: 0, head: 0, threshold: 0.05, positions: [],
               points: null, lines: null, particles: null, flows: [], edges: [], hovered: null };

function ringPositions(n) {
  const r = n > 16 ? 9 : 7;
  return Array.from({ length: n }, (_, i) => {
    const a = Math.PI / 2 - (2 * Math.PI * i) / n;
    return new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), 0);
  });
}

function attnCurve(from, to) {
  const mid = from.clone().add(to).multiplyScalar(0.5);
  const lift = from.distanceTo(to) * 0.45;
  mid.z += lift;
  return new THREE.QuadraticBezierCurve3(from, mid, to);
}

function buildAttention() {
  const { view, data, block, head, threshold } = attn;
  for (const obj of [attn.points, attn.lines, attn.particles]) disposeObject(obj);
  const tokens = data.tokens;
  const weights = data.attention[block][head];
  attn.positions = ringPositions(tokens.length);
  attn.points = pointsObject(attn.positions.flatMap((v) => v.toArray()), 1.1);
  view.scene.add(attn.points);

  // Causal edges only: query r attends to key c with c < r. Self-attention (c === r)
  // is reported in the hover panel rather than drawn as a line.
  attn.edges = [];
  let total = 0;
  for (let r = 0; r < tokens.length; r++) {
    for (let c = 0; c < r; c++) {
      total++;
      const w = weights[r][c];
      if (w >= threshold && w > 0) attn.edges.push({ query: r, key: c, weight: w });
    }
  }
  attn.flows = attn.edges.map((e) => {
    const n = 1 + Math.round(e.weight * 3);
    return { ...e, curve: attnCurve(attn.positions[e.key], attn.positions[e.query]),
             phase: Array.from({ length: n }, (_, k) => k / n) };
  });
  const particleCount = attn.flows.reduce((s, f) => s + f.phase.length, 0);
  attn.particles = pointsObject(new Array(particleCount * 3).fill(0), 0.5);
  view.scene.add(attn.particles);
  attn.lines = new THREE.Group();
  view.scene.add(attn.lines);
  colourAttention();

  const hidden = total - attn.edges.length;
  const why = threshold > 0 ? `have a weight below ${threshold.toFixed(2)}`
    : "have a weight of 0 at the 6 decimal places the server sends";
  $("edge_count").textContent =
    `Block ${block + 1}, head ${head + 1}: showing ${attn.edges.length} of ${total} lines from ` +
    `tokens to earlier tokens.${hidden ? ` The other ${hidden} ${why}.` : ""} ` +
    "Self-attention is listed on hover.";
  debug.attn = {
    block, head, threshold, tokens,
    edges: attn.edges.map((e) => ({ query: e.query, key: e.key, weight: e.weight })),
    total,
  };
}

function colourAttention() {
  const { hovered } = attn;
  const colors = attn.points.geometry.getAttribute("color");
  attn.positions.forEach((_, i) => {
    const c = i === hovered ? NODE_HOT : NODE.clone().multiplyScalar(hovered === null ? 1 : 0.6);
    colors.setXYZ(i, c.r, c.g, c.b);
  });
  colors.needsUpdate = true;
  const positions = [];
  const lineColors = [];
  for (const f of attn.flows) {
    const touches = hovered !== null && f.query === hovered;
    const dim = hovered === null || touches ? 1 : 0.2;
    const color = (touches ? EDGE_HOT : EDGE).clone().multiplyScalar(f.weight * dim);
    pushSegmentPath(f.curve.getPoints(24), color, positions, lineColors);
  }
  for (const child of [...attn.lines.children]) disposeObject(child);
  attn.lines.add(lineObject(positions, lineColors));
}

function animateParticles(time) {
  if (!attn.particles) return;
  const pos = attn.particles.geometry.getAttribute("position");
  const col = attn.particles.geometry.getAttribute("color");
  let k = 0;
  for (const f of attn.flows) {
    const speed = 0.08 + 0.5 * f.weight;
    const dim = attn.hovered === null || f.query === attn.hovered ? 1 : 0.15;
    const c = PARTICLE.clone().multiplyScalar(Math.min(1, 0.25 + f.weight) * dim);
    for (const phase of f.phase) {
      const t = (phase + time * 0.001 * speed) % 1;
      const p = f.curve.getPoint(t);
      pos.setXYZ(k, p.x, p.y, p.z);
      col.setXYZ(k, c.r, c.g, c.b);
      k++;
    }
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
}

function attnLabels() {
  const { view, data, hovered } = attn;
  if (!data) return;
  const weights = data.attention[attn.block][attn.head];
  placeLabels(view, data.tokens.map((tok, i) => {
    const parts = [{ text: shown(tok) }];
    let cls = hovered === null ? "" : i === hovered ? "main" : "dim";
    if (hovered !== null && i < hovered && weights[hovered][i] >= attn.threshold) {
      parts.push({ text: ` ${fmtNum(weights[hovered][i])}`, num: true });
      cls = "";
    }
    return { position: attn.positions[i], parts, cls };
  }));
}

function showAttentionInfo(i) {
  const { data } = attn;
  const info = $("attn_info");
  info.replaceChildren();
  if (i === null) {
    info.textContent = "Hover over a token to see what it attends to.";
    return;
  }
  const weights = data.attention[attn.block][attn.head][i];
  const h = document.createElement("h3");
  h.textContent = `"${shown(data.tokens[i])}" attends to`;
  const table = document.createElement("table");
  weights.slice(0, i + 1)
    .map((w, c) => [c, w])
    .sort((a, b) => b[1] - a[1])
    .forEach(([c, w]) => {
      const tr = table.insertRow();
      tr.insertCell().textContent = c === i ? `${shown(data.tokens[c])} (itself)` : shown(data.tokens[c]);
      const v = tr.insertCell();
      v.className = "v";
      v.textContent = fmtNum(w);
    });
  const note = document.createElement("div");
  note.className = "sub";
  note.textContent = `Block ${attn.block + 1}, head ${attn.head + 1}. Weights sum to 1.`;
  info.append(h, table, note);
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
    button.setAttribute("aria-label", `${label} ${i + 1}`);
    button.textContent = String(i + 1);
    button.addEventListener("click", () => onPick(i));
    holder.appendChild(button);
  }
}

function selectAttention(block, head) {
  attn.block = block;
  attn.head = head;
  attn.hovered = null;
  renderPicker("block_picker", "Block", attn.data.num_blocks, block, (b) => selectAttention(b, attn.head));
  renderPicker("head_picker", "Head", attn.data.num_heads, head, (h) => selectAttention(attn.block, h));
  buildAttention();
  showAttentionInfo(null);
}

async function loadAttention(event) {
  if (event) event.preventDefault();
  const errorBox = $("attn_error");
  errorBox.hidden = true;
  const prompt = $("attn_prompt").value;
  if (!prompt.trim()) {
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
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      errorBox.textContent = data.error || `The server returned an error (${response.status}).`;
      errorBox.hidden = false;
      return;
    }
    if (!data.verified) throw new Error("the server did not verify these weights");
    attn.data = data;
    selectAttention(Math.min(attn.block, data.num_blocks - 1), Math.min(attn.head, data.num_heads - 1));
  } catch (err) {
    errorBox.textContent = `Could not load attention: ${err.message}`;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
}

function initAttention() {
  attn.view = makeStage("stage_attn", 24);
  attn.view.camera.position.set(0, -7, 23);
  const { view } = attn;
  view.onFrame = attnLabels;
  const canvas = view.renderer.domElement;
  const onPointer = (event) => {
    if (!attn.points) return;
    const i = hoverIndex(view, attn.points, event, 0.6);
    if (i === attn.hovered) return;
    attn.hovered = i;
    colourAttention();
    showAttentionInfo(i);
  };
  canvas.addEventListener("pointermove", onPointer);
  canvas.addEventListener("pointerdown", onPointer);
  $("attn_form").addEventListener("submit", loadAttention);
  $("threshold").addEventListener("input", (e) => {
    attn.threshold = Number(e.target.value);
    $("threshold_value").textContent = attn.threshold.toFixed(2);
    if (attn.data) buildAttention();
  });
  $("rotate_attn").addEventListener("change", (e) => { view.wantsRotate = e.target.checked; });
  $("attn_legend").style.background = LEGEND;
}

// --- tabs, loop --------------------------------------------------------------------------

function selectTab(name) {
  const embedOn = name === "embed";
  $("tab_embed").setAttribute("aria-selected", String(embedOn));
  $("tab_attn").setAttribute("aria-selected", String(!embedOn));
  $("panel_embed").hidden = !embedOn;
  $("panel_attn").hidden = embedOn;
  embed.view.active = embedOn;
  attn.view.active = !embedOn;
  if (!embedOn && !attn.data) loadAttention();
}

function loop(time) {
  for (const view of [embed.view, attn.view]) {
    if (!view.active) continue;
    view.controls.autoRotate = view.wantsRotate && !view.interacting;
    view.controls.update();
    if (view === attn.view) animateParticles(time);
    view.renderer.render(view.scene, view.camera);
    if (view.onFrame) view.onFrame();
  }
  requestAnimationFrame(loop);
}

function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch (err) {
    return false;
  }
}

if (!webglAvailable()) {
  $("webgl_error").hidden = false;
  $("panel_embed").hidden = true;
  $("panel_attn").hidden = true;
} else {
  initEmbedding();
  initAttention();
  debug.camera = (which) => (which === "attn" ? attn : embed).view.camera.position.toArray();
  debug.screenOf = (which, i) => {
    const state = which === "attn" ? attn : embed;
    return toScreen(state.view, state.positions[i]);
  };
  $("tab_embed").addEventListener("click", () => selectTab("embed"));
  $("tab_attn").addEventListener("click", () => selectTab("attn"));
  requestAnimationFrame(loop);
}
