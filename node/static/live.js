// Shared pieces: one poller of /api/live and the live-telemetry renderer (Replay, Walkthrough). The ingest-health and
// raw-batch renderers and the recorded-trace player went with the Debug page, 2026-11-27.
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const style = document.createElement("style");
  style.textContent = `
    .host h2 { font-size: 15px; font-weight: 600; margin: 0 0 8px; display: flex; gap: 12px; align-items: baseline; }
    .strip { margin-bottom: 10px; }
    .legend { display: flex; gap: 14px; margin: 0 0 8px; font-size: 12.5px; color: var(--ink2); }
    .legend i { display: inline-block; width: 14px; height: 2px; margin-right: 6px; vertical-align: middle; }
    .strip .lbl { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--ink2); }
    .strip .lbl b { color: var(--ink); font-weight: 600; }
    .chart { position: relative; } .chart canvas { width: 100%; display: block; }
    .tip { position: absolute; pointer-events: none; background: var(--ink); color: var(--surface);
           font-size: 12px; padding: 2px 6px; border-radius: 4px; display: none; white-space: nowrap; }
    .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
    .stat .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .stat .k { font-size: 12px; color: var(--ink2); }
    .fields { margin-top: 8px; font-size: 12px; color: var(--ink2); }
    .host h2 { flex-wrap: wrap; row-gap: 4px; }
    .pill { white-space: nowrap; font-size: 11.5px; font-weight: 400; padding: 1px 9px; border-radius: 999px; background: var(--flash); color: var(--ink); }
    .act { font: inherit; font-size: 12.5px; font-weight: 400; padding: 3px 12px; border: 1px solid var(--ring); border-radius: 6px;
           background: var(--page); color: var(--ink); cursor: pointer; text-decoration: none; }
    .act:hover { background: var(--grid); }
    .nodata { display: flex; gap: 10px; align-items: center; margin: 0 0 10px; }
    .host h2 .grow { flex: 1; }
    .boxes { display: grid; gap: 2px; margin: 0 0 6px; }
    .box { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; row-gap: 2px; font-size: 13.5px; padding: 3px 0; }
    .box.off { opacity: .45; }
    .box .grow { flex: 1; }
    .tog { all: unset; cursor: pointer; display: inline-flex; gap: 8px; align-items: center; }
    .tog .sw { width: 12px; height: 12px; border-radius: 3px; border: 2px solid; box-sizing: border-box; }
    .tog b { font-weight: 600; } .tog .lab { color: var(--ink2); }
    .hintrow { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0 10px; }
    .hintrow .grow { flex: 1; } .hintrow .acts { display: flex; gap: 8px; }
    .hint { font-size: 12px; color: var(--muted); margin: 0; }
    .act:disabled { opacity: .5; cursor: not-allowed; } .act:disabled:hover { background: var(--page); }
    .strip .lbl .now > span + span { margin-left: 12px; }
    /* compact: the same three strips at the top of another page (Replay demo). One row of boxes, no cards. */
    .compact .boxes { display: flex; flex-wrap: wrap; gap: 2px 20px; align-items: center; margin: 4px 0 0; }
    .compact .box { padding: 0; font-size: 13px; } .compact .box .grow { display: none; }
    .compact .nodata { margin: 0; }
    .compact .strip { margin: 8px 0 0; }
  `;
  document.head.appendChild(style);

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // up to MAX_BOXES boxes share the graphs; each keeps the colour of the slot the node gave it
  const MAX_BOXES = 8;   // = MAX_BOXES in node/app.py
  const BOX_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#7f5af0", "#d6a419", "#d6467e", "#2fb5c9", "#8a6d4a"];
  const colorOf = h => h.color || BOX_COLORS[(h.slot || 0) % BOX_COLORS.length];   // a page may set each node's colour (Replay: by workload)
  const alpha = (hex, a) => hex + Math.round(a * 255).toString(16).padStart(2, "0");
  const hidden = new Set(JSON.parse(localStorage.getItem("twc-hidden-boxes") || "[]"));   // boxes switched off on the graphs
  const saveHidden = () => localStorage.setItem("twc-hidden-boxes", JSON.stringify([...hidden]));
  let liveView = null;   // {el, data}: lets a toggle redraw at once instead of on the next poll
  const bySlot = data => Object.keys(data).sort((a, b) => data[a].slot - data[b].slot);

  // box toggles: delegated, since the box rows are rebuilt when boxes come and go
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-tog]"); if (!t) return;
    const x = t.dataset.tog; hidden.has(x) ? hidden.delete(x) : hidden.add(x); saveHidden();
    if (liveView) renderLive(liveView.el, liveView.data);
  });
  // connectHref: where "Connect a box" goes; the Walkthrough's section 1 installs the collector (Debug's own Connect page went, 2026-11-22)
  const noDataBarFor = connectHref => `<div class="nodata"><a class="act" href="${connectHref}">Connect a box</a></div>`;
  const noDataBar = noDataBarFor("/walkthrough");

  // short: just the state (compact view), without the batch count
  const statusLine = (h, short) => {
    if (!h.batches) return `<span class="dot" style="background:${css("--muted")}"></span>` + (short ? "waiting for its first batch" : "said hello · waiting for its first batch");
    const live = h.age_s < 3;
    return `<span class="dot" style="background:${live ? css("--good") : css("--crit")}"></span>` +
      (live ? "streaming" : "silent for " + Math.round(h.age_s) + " s") + (short ? "" : ` · ${h.batches} batches received`) +
      Object.entries(h.double_counted || {}).map(([o, n]) => ` · <b style="color:${css("--crit")}">${n} GPU(s) also reported by ${esc(o)}</b>`).join("");
  };
  // ---- renderer 1: live telemetry. Three stacked strips sharing one 60 s axis (power, HBM bandwidth,
  // NVLink). Per box: a thick line = mean across its GPUs, faint thin lines = each GPU, in the box's colour.
  // Each box is drawn against its own clock ("seconds ago" from its newest sample), so clock skew
  // between boxes doesn't slide their lines apart.
  const nice = v => { const m = Math.max(v, 1), e = 10 ** Math.floor(Math.log10(m)); return [1, 2, 5, 10].map(k => k * e).find(x => x >= m); };
  // hc / ticksC: canvas height and gridlines in the compact view (about half the height, fewer gridlines)
  const STRIPS = [
    { key: "power", label: "Power", h: 150, hc: 76, max: () => 800, ticks: [0, 200, 400, 600, 800], ticksC: [0, 400, 800], unit: "W", fmt: v => v.toFixed(0) + " W" },
    { key: "hbm_bw", label: "HBM bandwidth used", h: 110, hc: 54, max: () => 100, ticks: [0, 50, 100], unit: "%", fmt: v => v.toFixed(1) + " %" },
    { key: "nvlink", label: "NVLink traffic (tx + rx)", h: 110, hc: 54, max: m => nice(m), unit: "GB/s", fmt: v => v.toFixed(2) + " GB/s" },
    // PCIe: GPU <-> host. Two strips because their floors differ: serving receives GB/s from host memory but sends little.
    { key: "pcie_tx", label: "PCIe sent by GPU", h: 90, hc: 46, max: m => nice(m), unit: "GB/s", fmt: v => v.toFixed(2) + " GB/s" },
    { key: "pcie_rx", label: "PCIe received by GPU", h: 90, hc: 46, max: m => nice(m), unit: "GB/s", fmt: v => v.toFixed(2) + " GB/s" },
  ];
  const STRIPS_COMPACT = STRIPS.map(st => ({ ...st, h: st.hc, ticks: st.ticksC || st.ticks }));
  // opts.strips: a page can pass its own strip list (the replay-first mock-up does); otherwise the built-in ones
  const stripsFor = o => o.strips || (o.compact ? STRIPS_COMPACT : STRIPS);
  const nearest = (pts, t) => { let b = null; for (const p of pts) if (!b || Math.abs(p[0] - t) < Math.abs(b[0] - t)) b = p; return b; };
  const lastT = s => s && s.length ? s[s.length - 1][0] : 0;
  // where "now" is for a box: its newest sample, pushed back by how long it has been silent
  const boxEnd = h => (lastT(h.mean.power) || lastT(h.mean.hbm_bw)) + (h.age_s > 3 ? h.age_s : 0);

  // o (the view's opts) can widen the time axis for a page that shows a whole recording at once:
  //   span      seconds the axis covers (default 60)        axis     [left label, right label]
  //   cursorAgo a playhead, in seconds before the right edge  marks    faint vertical lines, same unit
  //   tipFmt    seconds before the right edge -> text for the hover card
  function drawStrip(cv, tip, st, data, hosts, hoverAgo, nAll, o = {}) {
    const span = o.span || 60;
    const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext("2d"); c.scale(dpr, dpr);
    const ends = Object.fromEntries(hosts.map(x => [x, boxEnd(data[x])]));
    const inWin = (x, pts) => pts.filter(p => p[0] >= ends[x] - span);
    const vmax = Math.max(0, ...hosts.flatMap(x => Object.values(data[x].gpus).flatMap(g => inWin(x, g[st.key]).map(p => p[1]))));
    const ymax = st.max(vmax), ticks = st.ticks || [0, ymax / 2, ymax];
    const L = 46, B = 16, X = ago => L + (W - L) * (1 - ago / span), y = v => 4 + (H - B - 4) * (1 - Math.min(v, ymax) / ymax);
    c.font = "11px ui-sans-serif, system-ui"; c.fillStyle = css("--muted"); c.strokeStyle = css("--grid"); c.lineWidth = 1;
    for (const v of ticks) {
      c.beginPath(); c.moveTo(L, Math.round(y(v)) + .5); c.lineTo(W, Math.round(y(v)) + .5); c.stroke();
      c.fillText((Number.isInteger(v) ? v : v.toFixed(1)) + " " + st.unit, 0, y(v) + 4);
    }
    if (o.axis) { c.fillText(o.axis[0], L, H - 2); c.fillText(o.axis[1], W - c.measureText(o.axis[1]).width - 1, H - 2); }
    else { c.fillText("−60 s", L, H - 2); c.fillText("now", W - 22, H - 2); }
    for (const a of o.marks || []) { c.setLineDash([2, 4]); c.strokeStyle = css("--axis"); c.beginPath();
      c.moveTo(Math.round(X(a)) + .5, 0); c.lineTo(Math.round(X(a)) + .5, H - B); c.stroke(); c.setLineDash([]); }
    cv._geo = { L, W };
    if (!hosts.length) {   // placeholder: same axes, no lines
      c.font = "12.5px ui-sans-serif, system-ui"; c.textAlign = "center";
      c.fillText(nAll ? "all boxes hidden" : "receiving no data right now", L + (W - L) / 2, (H - B) / 2 + 4); c.textAlign = "start";
      tip.style.display = "none"; return;
    }
    const line = (x, pts, color, width) => {
      c.strokeStyle = color; c.lineWidth = width; c.lineJoin = "round"; c.beginPath();
      inWin(x, pts).forEach(([t, v], k) => k ? c.lineTo(X(ends[x] - t), y(v)) : c.moveTo(X(ends[x] - t), y(v)));
      c.stroke();
    };
    for (const x of hosts) {   // faint per-GPU lines first, so every box's mean sits on top
      const gs = Object.values(data[x].gpus);
      // a strip can supply its own colour for the per-GPU lines (st.gpuColor: box colour -> CSS colour)
      const faint = st.gpuColor ? st.gpuColor(colorOf(data[x])) : alpha(colorOf(data[x]), .28);
      if (gs.length > 1) gs.forEach(g => line(x, g[st.key], faint, 1));
    }
    for (const x of hosts) line(x, data[x].mean[st.key], colorOf(data[x]), o.span ? 1.6 : 2.2);
    if (o.cursorAgo != null) { c.strokeStyle = css("--crit"); c.lineWidth = 1.2; c.beginPath();
      c.moveTo(Math.round(X(o.cursorAgo)) + .5, 0); c.lineTo(Math.round(X(o.cursorAgo)) + .5, H - B); c.stroke(); }
    if (hoverAgo == null) { tip.style.display = "none"; return; }
    c.strokeStyle = css("--axis"); c.lineWidth = 1; c.beginPath();
    c.moveTo(Math.round(X(hoverAgo)) + .5, 0); c.lineTo(Math.round(X(hoverAgo)) + .5, H - B); c.stroke();
    const parts = hosts.map(x => { const p = nearest(data[x].mean[st.key], ends[x] - hoverAgo);
      return p ? `<span style="color:${colorOf(data[x])}">●</span> ${st.fmt(p[1])}` : null; }).filter(Boolean);
    tip.style.display = "block"; tip.innerHTML = parts.join(" &nbsp;") + " · " + (o.tipFmt ? o.tipFmt(hoverAgo) : `${hoverAgo.toFixed(1)} s ago`);
    tip.style.left = Math.min(X(hoverAgo) + 8, W - tip.offsetWidth - 4) + "px"; tip.style.top = "2px";
  }

  // opts (optional, remembered on el): { compact: true, connectHref: "/walkthrough" } for the short
  // version other pages mount. Without opts this is the Debug page's view, unchanged.
  function renderLive(el, data, opts) {
    if (opts) el._opts = opts;
    const o = el._opts || {}, strips = stripsFor(o);
    liveView = { el, data };
    if (!el.dataset.built) {   // the strips are built once; boxes only change the rows above them
      el.dataset.built = 1;
      el.classList.toggle("compact", !!o.compact);
      el.innerHTML = `<div class="boxes"></div>` + (o.compact ? "" : `
        <div class="hintrow"><p class="hint">Thick line: mean across a box's GPUs · faint lines: each GPU · click a box to hide or show it</p>
          <span class="grow"></span><span class="acts"></span></div>`) +
        strips.map(st => `<div class="${o.compact ? "" : "card "}strip" data-k="${st.key}"><div class="lbl"><span>${st.label}</span><span class="now"></span></div>
          <div class="chart"><canvas style="height:${st.h}px"></canvas><div class="tip"></div></div></div>`).join("");
      el.querySelectorAll(".strip canvas").forEach(cv => {
        cv.addEventListener("mousemove", e => {
          const g = cv._geo; if (!g) return;
          const span = (el._opts || {}).span || 60, ago = span * (1 - (e.clientX - cv.getBoundingClientRect().left - g.L) / (g.W - g.L));
          el._hoverAgo = ago < 0 || ago > span ? null : ago; drawAll(el);
        });
        cv.addEventListener("mouseleave", () => { el._hoverAgo = null; drawAll(el); });
      });
    }
    const hosts = bySlot(data), boxes = el.querySelector(".boxes");
    const sig = hosts.map(x => [x, data[x].slot, data[x].label, data[x].replay, data[x].stoppable, hidden.has(x), data[x].age_s > 60].join("|")).join(";") || "none";
    if (boxes.dataset.sig !== sig) {   // rebuild rows only when the set of boxes changes (keeps buttons clickable)
      boxes.dataset.sig = sig;
      // a box silent for over 60 s gives up its slot to a newcomer, so it doesn't block adding one
      boxes.innerHTML = !hosts.length
        ? `<div class="box"><b>No box connected</b><span class="sub"><span class="dot" style="background:${css("--muted")}"></span>receiving no data right now</span></div>${o.connectHref ? noDataBarFor(o.connectHref) : noDataBar}`
        : hosts.map(x => {
            const h = data[x], col = colorOf(h), off = hidden.has(x);
            return `<div class="box${off ? " off" : ""}" data-host="${esc(x)}">
              <button class="tog" data-tog="${esc(x)}" title="${off ? "Show" : "Hide"} this box on the graphs">
                <i class="sw" style="border-color:${col};background:${off ? "transparent" : col}"></i><b>${esc(x)}</b><span class="lab">${esc(h.label)}</span></button>
              <span class="sub st"></span><span class="grow"></span></div>`;
          }).join("");
      // beside the hint: Connect another box, while boxes are on the page and a slot is free (the empty state has its own)
      const room = hosts.filter(x => data[x].age_s <= 60).length < MAX_BOXES;
      const acts = el.querySelector(".acts");   // the compact view has no hint row, so no buttons beside it
      if (acts) acts.innerHTML = hosts.length && room ? `<a class="act" href="/walkthrough">Connect another box</a>` : "";
    }
    for (const x of hosts) {
      const st = boxes.querySelector(`.box[data-host="${CSS.escape(x)}"] .st`);
      if (st) st.innerHTML = statusLine(data[x], o.compact) + (data[x].about && !o.compact ? " · " + esc(data[x].about) : "");
    }
    drawAll(el);
  }

  function drawAll(el) {
    const data = liveView ? liveView.data : {}, o = el._opts || {};
    const vis = bySlot(data).filter(x => !hidden.has(x) && Object.keys(data[x].gpus).length);
    for (const st of stripsFor(el._opts || {})) {
      const card = el.querySelector(`.strip[data-k="${st.key}"]`);
      card.querySelector(".now").innerHTML = vis.map(x => { const m = data[x].mean[st.key];   // current values: streaming boxes only
        const p = o.cursorAgo != null ? nearest(m, lastT(m) - o.cursorAgo) : m[m.length - 1];   // with a playhead: the value under it
        return m.length && data[x].age_s < 3 ? `<span><span style="color:${colorOf(data[x])}">●</span> <b>${st.fmt(p[1])}</b></span>` : ""; }).join("") || "–";
      drawStrip(card.querySelector("canvas"), card.querySelector(".tip"), st, data, vis, el._hoverAgo, Object.keys(data).length, o);
    }
  }

  // ---- one poller, many listeners
  const listeners = [];
  async function poll() {
    try { const d = await (await fetch("/api/live")).json(); listeners.forEach(f => f(d)); } catch (e) {}
  }
  window.TWCLive = {
    renderLive,
    subscribe(f) { listeners.push(f); if (listeners.length === 1) { poll(); setInterval(poll, 1000); } },
  };
})();
