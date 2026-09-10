const TAB20 = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
  "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78",
  "#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7"
];
const MAGMA = ["#000004","#180f3d","#4b1d73","#8c2981","#c63e73","#fb8861","#f6d746"];
const state = {
  meta: null, ds: null, mode: "click", color: "published", slide: 0,
  selected: [], box: null, dragging: false, view: null, vis: [],
};

function parseBin(buf) {
  const n = new DataView(buf).getUint32(0, true);
  const k = new DataView(buf).getUint32(4, true);
  let off = 8;
  const f32 = (count) => {
    const a = new Float32Array(buf.slice(off, off + count * 4));
    off += count * 4;
    return a;
  };
  const x = f32(n), y = f32(n), entropy = f32(n), vacuity = f32(n), p1 = f32(n);
  const published = new Uint8Array(buf.slice(off, off + n)); off += n;
  const pred = new Uint8Array(buf.slice(off, off + n)); off += n;
  const slide = new Uint8Array(buf.slice(off, off + n)); off += n;
  const p = new Float32Array(buf.slice(off, off + n * k * 4));
  return { n, k, x, y, entropy, vacuity, p1, published, pred, slide, p };
}

function visible() {
  const ds = state.ds, out = [];
  for (let i = 0; i < ds.n; i++) if (ds.slide[i] === state.slide) out.push(i);
  return out;
}

function distOne(ds, i) {
  const p = new Float64Array(ds.k);
  const row = i * ds.k;
  for (let k = 0; k < ds.k; k++) p[k] = ds.p[row + k];
  return p;
}

function distMean(ds, idx) {
  const p = new Float64Array(ds.k);
  for (const i of idx) {
    const row = i * ds.k;
    for (let k = 0; k < ds.k; k++) p[k] += ds.p[row + k];
  }
  for (let k = 0; k < ds.k; k++) p[k] /= idx.length;
  return p;
}

function lerpColor(t) {
  t = Math.min(1, Math.max(0, t));
  const u = t * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, Math.floor(u));
  const f = u - i;
  const hex = (s, j) => parseInt(s.slice(1 + j * 2, 3 + j * 2), 16);
  const mix = (c) => Math.round(hex(MAGMA[i], c) * (1 - f) + hex(MAGMA[i + 1], c) * f);
  return `rgb(${mix(0)},${mix(1)},${mix(2)})`;
}

function layout(ids, w, h, pad) {
  const ds = state.ds;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const i of ids) {
    if (ds.x[i] < minX) minX = ds.x[i]; if (ds.x[i] > maxX) maxX = ds.x[i];
    if (ds.y[i] < minY) minY = ds.y[i]; if (ds.y[i] > maxY) maxY = ds.y[i];
  }
  const s = Math.min((w - 2 * pad) / Math.max(maxX - minX, 1e-6), (h - 2 * pad) / Math.max(maxY - minY, 1e-6));
  return { minX, maxX, minY, maxY, s, ox: (w - s * (maxX - minX)) / 2, oy: (h - s * (maxY - minY)) / 2 };
}

function toPx(view, x, y) {
  return [view.ox + (x - view.minX) * view.s, view.oy + (view.maxY - y) * view.s];
}

function cellColor(ds, i) {
  const mode = state.color;
  if (mode === "published") return TAB20[ds.published[i] % TAB20.length];
  if (mode === "pred") return TAB20[ds.pred[i] % TAB20.length];
  if (mode === "agree") return ds.published[i] === ds.pred[i] ? "#2ca02c" : "#d62728";
  if (mode === "entropy") return lerpColor((ds.entropy[i] - 0.15) / 1.2);
  return lerpColor((ds.vacuity[i] - 0.03) / 0.06);
}

function drawLegend() {
  const el = document.getElementById("legend");
  const ds = state.ds;
  if (state.color === "published" || state.color === "pred") {
    el.innerHTML = ds.types.map((t, i) =>
      `<span><i class="swatch" style="background:${TAB20[i % TAB20.length]}"></i>${t}</span>`
    ).join("");
    return;
  }
  if (state.color === "agree") {
    el.innerHTML = `<span><i class="swatch" style="background:#2ca02c"></i>agrees</span><span><i class="swatch" style="background:#d62728"></i>disagrees</span>`;
    return;
  }
  el.innerHTML = `<span>low ${state.color}</span><span style="flex:1;height:8px;border-radius:99px;background:linear-gradient(90deg,#000004,#8c2981,#f6d746)"></span><span>high</span>`;
}

function drawMap() {
  const canvas = document.getElementById("map");
  const wrap = document.getElementById("map-wrap");
  const ds = state.ds;
  if (!ds) return;
  state.vis = visible();
  const cssW = wrap.clientWidth || 640;
  const cssH = Math.max(440, Math.round(cssW * 0.72));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  const overlay = document.getElementById("overlay");
  overlay.width = canvas.width;
  overlay.height = canvas.height;
  overlay.style.width = cssW + "px";
  overlay.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#f3efe6";
  ctx.fillRect(0, 0, cssW, cssH);
  const view = layout(state.vis, cssW, cssH, 16);
  state.view = { ...view, cssW, cssH, dpr };
  if (!state.vis.length || !isFinite(view.s)) {
    drawOverlay();
    drawLegend();
    return;
  }
  const chosen = new Set(state.selected);
  for (const i of state.vis) {
    const [px, py] = toPx(view, ds.x[i], ds.y[i]);
    ctx.fillStyle = cellColor(ds, i);
    ctx.beginPath();
    ctx.arc(px, py, 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 1.5;
  for (const i of chosen) {
    if (ds.slide[i] !== state.slide) continue;
    const [px, py] = toPx(view, ds.x[i], ds.y[i]);
    ctx.beginPath();
    ctx.arc(px, py, 4.8, 0, Math.PI * 2);
    ctx.stroke();
  }
  drawOverlay();
  drawLegend();
}

function drawOverlay() {
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const dpr = state.view ? state.view.dpr : 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!state.box) return;
  const { x0, y0, x1, y1 } = state.box;
  ctx.fillStyle = "rgba(35,78,112,.12)";
  ctx.strokeStyle = "rgba(27,27,24,.85)";
  ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
}

function eventPos(ev) {
  const r = document.getElementById("map").getBoundingClientRect();
  return [ev.clientX - r.left, ev.clientY - r.top];
}

function nearest(px, py) {
  const ds = state.ds, view = state.view;
  let best = -1, bestD = 45 * 45;
  for (const i of state.vis) {
    const [x, y] = toPx(view, ds.x[i], ds.y[i]);
    const d = (x - px) ** 2 + (y - py) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function inBox(px0, py0, px1, py1) {
  const ds = state.ds, view = state.view;
  const xLo = Math.min(px0, px1), xHi = Math.max(px0, px1);
  const yLo = Math.min(py0, py1), yHi = Math.max(py0, py1);
  const out = [];
  for (const i of state.vis) {
    const [x, y] = toPx(view, ds.x[i], ds.y[i]);
    if (x >= xLo && x <= xHi && y >= yLo && y <= yHi) out.push(i);
  }
  return out;
}

function renderBars(p, pubIdx, title, meta) {
  document.getElementById("panel-title").textContent = title;
  document.getElementById("panel-meta").textContent = meta;
  const rows = state.ds.types.map((name, k) => ({ name, v: p[k], k }));
  rows.sort((a, b) => b.v - a.v);
  const max = Math.max(rows[0] ? rows[0].v : 1, 1e-6);
  document.getElementById("bars").innerHTML = rows.map(r => `
    <div class="bar-row ${r.k === pubIdx ? "published" : ""}">
      <span class="name" title="${r.name}">${r.name}${r.k === pubIdx ? " · published" : ""}</span>
      <div class="track"><div class="fill" style="width:${(100 * r.v / max).toFixed(1)}%;background:${TAB20[r.k % TAB20.length]}"></div></div>
      <span class="pct">${(100 * r.v).toFixed(1)}</span>
    </div>
  `).join("");
}

function showSelection() {
  const ds = state.ds;
  const idx = state.selected.filter(i => ds.slide[i] === state.slide);
  state.selected = idx;
  if (!idx.length) {
    document.getElementById("panel-title").textContent = "No cell selected";
    document.getElementById("panel-meta").textContent = "Click a well, or box-select a neighborhood.";
    document.getElementById("bars").innerHTML = '<p class="empty">The label distribution appears here.</p>';
    drawMap();
    document.getElementById("status").textContent = `${state.vis.length} wells on ${state.meta.slides[state.slide]}`;
    return;
  }
  if (idx.length === 1) {
    const i = idx[0];
    const pub = ds.types[ds.published[i]];
    const pred = ds.types[ds.pred[i]];
    renderBars(
      distOne(ds, i),
      ds.published[i],
      pub,
      `${state.meta.slides[ds.slide[i]]} · pred ${pred}${pub === pred ? "" : " (disagrees)"} · H=${ds.entropy[i].toFixed(3)} · u=${ds.vacuity[i].toFixed(3)} · p₁=${ds.p1[i].toFixed(3)}`
    );
    document.getElementById("status").textContent = `1 cell · ${pub}`;
  } else {
    renderBars(distMean(ds, idx), -1, `Mean of ${idx.length} cells`, "Average simplex over the selection.");
    document.getElementById("status").textContent = `${idx.length} cells selected`;
  }
  drawMap();
}

function fillMeta(meta) {
  const m = meta.metrics;
  const acc = m.accuracy != null ? m.accuracy : m.oof_accuracy;
  document.getElementById("chips").innerHTML = [
    `${meta.n.toLocaleString()} wells · ${meta.k} types`,
    `edge ${m.n_edge} · core ${m.n_core}`,
    `accuracy ${acc.toFixed(3)}`,
    `mean entropy ${m.mean_entropy.toFixed(3)}`,
    `mean p₁ ${m.mean_p1.toFixed(3)}`,
    `mean vacuity ${m.mean_vacuity.toFixed(3)}`,
  ].map(t => `<span class="chip">${t}</span>`).join("");
  const rows = Object.entries(meta.per_type).sort((a, b) => b[1].n - a[1].n);
  document.getElementById("type-table").innerHTML = rows.map(([name, r]) => `
    <tr>
      <td>${name}</td>
      <td class="num">${r.n}</td>
      <td class="num">${r.accuracy.toFixed(3)}</td>
      <td class="num">${r.mean_entropy.toFixed(3)}</td>
      <td class="num">${r.mean_p1.toFixed(3)}</td>
      <td class="num">${r.mean_vacuity.toFixed(3)}</td>
    </tr>
  `).join("");
}

function bind() {
  const surface = document.getElementById("map-wrap");
  document.getElementById("slide").addEventListener("change", ev => {
    state.slide = Number(ev.target.value);
    const vis = visible();
    state.selected = vis.length ? [vis[Math.floor(vis.length / 2)]] : [];
    showSelection();
  });
  document.getElementById("color").addEventListener("change", ev => {
    state.color = ev.target.value;
    drawMap();
  });
  document.getElementById("mode-click").addEventListener("click", () => {
    state.mode = "click";
    document.getElementById("mode-click").classList.add("active");
    document.getElementById("mode-box").classList.remove("active");
  });
  document.getElementById("mode-box").addEventListener("click", () => {
    state.mode = "box";
    document.getElementById("mode-box").classList.add("active");
    document.getElementById("mode-click").classList.remove("active");
  });
  document.getElementById("clear").addEventListener("click", () => {
    state.selected = [];
    state.box = null;
    showSelection();
  });
  surface.addEventListener("pointerdown", ev => {
    const [px, py] = eventPos(ev);
    if (state.mode === "click") {
      const i = nearest(px, py);
      state.selected = i >= 0 ? [i] : [];
      showSelection();
      return;
    }
    state.dragging = true;
    state.box = { x0: px, y0: py, x1: px, y1: py };
    surface.setPointerCapture(ev.pointerId);
    drawOverlay();
  });
  surface.addEventListener("pointermove", ev => {
    if (!state.dragging || !state.box) return;
    const [px, py] = eventPos(ev);
    state.box.x1 = px; state.box.y1 = py;
    drawOverlay();
  });
  surface.addEventListener("pointerup", () => {
    if (!state.dragging || !state.box) return;
    state.dragging = false;
    const b = state.box;
    state.selected = inBox(b.x0, b.y0, b.x1, b.y1);
    state.box = null;
    showSelection();
  });
  window.addEventListener("resize", () => { if (state.ds) drawMap(); });
}

fetch("explorer.json")
  .then(r => r.json())
  .then(async meta => {
    state.meta = meta;
    fillMeta(meta);
    const buf = await fetch(meta.file).then(r => r.arrayBuffer());
    state.ds = parseBin(buf);
    state.ds.types = meta.types;
    bind();
    const vis = visible();
    state.selected = vis.length ? [vis[Math.floor(vis.length / 2)]] : [];
    showSelection();
  })
  .catch(err => {
    document.getElementById("status").textContent = "Could not load explorer data.";
    console.error(err);
  });
