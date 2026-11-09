"use strict";

// Pictures for the walkthrough steps. The server sends a small JSON spec
// ({kind: ...}) next to the text; render(spec) turns it into a DOM element.
// Everything is built with createElementNS / textContent, never innerHTML.

const Figures = (() => {
  const SVG = "http://www.w3.org/2000/svg";
  const PETS = "#7ee0a8";     // dot colour for pet words
  const BAKING = "#ffb454";   // dot colour for baking words

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, content, cls, attrs) {
    const node = svg("text", { x, y, class: cls || "", ...(attrs || {}) }, parent);
    node.textContent = content;
    return node;
  }

  function html(tag, attrs, parent, content) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (content !== undefined) node.textContent = content;
    if (parent) parent.appendChild(node);
    return node;
  }

  function frame(caption, width, height) {
    const box = html("figure", { class: "fig" });
    const canvas = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img",
                                "aria-label": caption, class: "figsvg" }, box);
    return { box, canvas };
  }

  function controls(box) {
    return html("div", { class: "figcontrols" }, box);
  }

  // Blue for positive values, orange for negative; brighter = bigger.
  function heat(value, vmax) {
    const level = Math.min(1, Math.abs(value) / vmax);
    const [r, g, b] = value >= 0 ? [58, 134, 255] : [255, 140, 60];
    return `rgba(${r},${g},${b},${(0.08 + 0.92 * level).toFixed(3)})`;
  }

  function matrix(parent, values, x, y, size, vmax, options = {}) {
    const group = svg("g", {}, parent);
    values.forEach((row, i) => row.forEach((value, j) => {
      svg("rect", { x: x + j * size, y: y + i * size, width: size - 1, height: size - 1,
                    rx: 2, fill: heat(value, vmax) }, group);
      if (options.numbers) {
        const shown = Math.abs(value) < 0.005 ? "0"
          : (Number.isInteger(value) ? String(value) : value.toFixed(options.digits || 1));
        text(group, x + j * size + size / 2, y + i * size + size / 2 + 4, shown,
             "cellnum", { "text-anchor": "middle" });
      }
    }));
    return group;
  }

  // ---- SVD on a table small enough to check by hand ----------------------

  function svdMini(spec) {
    const { box, canvas } = frame("A small table rebuilt from two layers", 720, 610);
    const size = 42;
    const vmax = spec.vmax || 4;
    const nm = spec.names || { column: "column", row: "row" };
    const labels = (parent, x, y, rows, cols) => {
      cols.forEach((c, j) => {
        const [first, second] = c.split(" ");
        if (second === undefined) {
          text(parent, x + j * size + size / 2, y - 6, c, "axs", { "text-anchor": "middle" });
        } else {   // "尾巴 tail": the word above its English
          text(parent, x + j * size + size / 2, y - 18, first, "axs", { "text-anchor": "middle" });
          text(parent, x + j * size + size / 2, y - 6, second, "axs", { "text-anchor": "middle" });
        }
      });
      rows.forEach((r, i) => text(parent, x - 8, y + i * size + size / 2 + 4, r, "axs",
                                  { "text-anchor": "end" }));
    };
    const numbers = (parent, values, x, y, vertical) => values.forEach((v, i) =>
      text(parent, vertical ? x + size / 2 : x + i * size + size / 2,
           vertical ? y + i * size + size / 2 + 4 : y + size / 2 + 4,
           v.toFixed(2), "cellnum", { "text-anchor": "middle" }));

    text(canvas, 20, 22, spec.title || "THE TABLE", "figtitle");
    labels(canvas, 100, 62, spec.rows, spec.cols);
    matrix(canvas, spec.table, 100, 62, size, vmax, { numbers: true });

    const blocks = spec.layers.map((layer, k) => {
      const top = 236 + k * 188;
      const g = svg("g", {}, canvas);
      text(g, 20, top - 22, `${k ? "+   " : ""}LAYER ${k + 1}  =  strength  ×  ${nm.column}  ×  ${nm.row}   (strength ${layer.strength.toFixed(2)})`, "figtitle");
      matrix(g, layer.column.map((v) => [v]), 100, top, size, 1);
      numbers(g, layer.column, 100, top, true);
      text(g, 100 + size / 2, top + 3 * size + 16, nm.column, "axs", { "text-anchor": "middle" });
      text(g, 168, top + 71, "×", "op");
      svg("circle", { cx: 232, cy: top + 63, r: 30, class: "strength" }, g);
      text(g, 232, top + 69, layer.strength.toFixed(2), "strengthnum", { "text-anchor": "middle" });
      text(g, 232, top + 3 * size + 16, "strength", "axs", { "text-anchor": "middle" });
      text(g, 286, top + 71, "×", "op");
      matrix(g, [layer.row], 330, top + size, size, 1);
      numbers(g, layer.row, 330, top + size, false);
      text(g, 330 + 1.5 * size, top + 3 * size + 16, nm.row, "axs", { "text-anchor": "middle" });
      text(g, 480, top + 71, "=", "op");
      labels(g, 590, top, spec.rows, spec.cols);
      matrix(g, layer.matrix, 590, top, size, vmax, { numbers: true, digits: spec.digits || 1 });
      return g;
    });
    const summary = text(canvas, 20, 598, spec.summary || "Layer 1 + layer 2 = the table again, cell for cell.",
                         "figtitle", { style: "display:none" });

    const two = blocks.length > 1;
    let both = false;
    const button = two ? html("button", { type: "button", class: "primary" }, controls(box), "Add layer 2") : null;
    const show = () => {
      if (two) {
        blocks[1].style.display = both ? "" : "none";
        summary.style.display = both ? "" : "none";
        button.textContent = both ? "Start over" : "Add layer 2";
      }
      canvas.setAttribute("viewBox", `0 0 720 ${both ? 610 : 420}`);
    };
    if (two) button.addEventListener("click", () => { both = !both; show(); });
    show();
    return box;
  }

  // ---- SVD on the real table: slide how many layers to keep --------------

  function svdReal(spec) {
    const n = spec.words.length;
    const cell = 10;
    const mapSize = n * cell;
    const wide = (word) => [...word].reduce((sum, ch) => sum + (ch.charCodeAt(0) > 0x2e80 ? 9 : 5.4), 0);
    const left = Math.max(70, Math.max(...spec.words.map(wide)) + 14), gap = 26;
    const width = left + 3 * mapSize + 2 * gap + 10;
    const { box, canvas } = frame("The table rebuilt from its strongest layers", width, 330);
    const mapY = 40;
    const xs = [left, left + mapSize + gap, left + 2 * (mapSize + gap)];

    ["THE TABLE", "REBUILT FROM k LAYERS", "WHAT IS LEFT OVER"].forEach((title, i) =>
      text(canvas, xs[i], 26, title, "figtitle"));
    const titleMid = canvas.querySelectorAll("text")[1];

    spec.words.forEach((word, i) => text(canvas, xs[0] - 5, mapY + i * cell + 8, word, "tiny",
                                         { "text-anchor": "end", style: `fill:${spec.pets[i] ? PETS : BAKING}` }));
    matrix(canvas, spec.table, xs[0], mapY, cell, 1.4);
    const rebuilt = svg("g", {}, canvas);
    const leftover = svg("g", {}, canvas);

    const chartY = 276;
    const bars = spec.strengths.map((value, i) => {
      const x = left + i * 30;
      const height = value / spec.strengths[0] * 52;
      const rect = svg("rect", { x, y: chartY - height, width: 22, height, rx: 2 }, canvas);
      text(canvas, x + 11, chartY - height - 4, value.toFixed(2), "tiny", { "text-anchor": "middle" });
      text(canvas, x + 11, chartY + 11, String(i + 1), "tiny", { "text-anchor": "middle" });
      return rect;
    });
    text(canvas, 10, chartY - 20, "layer", "tiny");
    text(canvas, 10, chartY - 8, "strength", "tiny");
    const legend = html("div", { class: "figlegend" }, box);
    html("span", { class: "dot pets" }, legend);
    legend.appendChild(document.createTextNode(" pet words   "));
    html("span", { class: "dot baking" }, legend);
    legend.appendChild(document.createTextNode(" baking words   (colours added by us for the picture; the table never had them)"));

    const bar = controls(box);
    html("label", { for: "svdk" }, bar, "Layers kept:");
    const slider = html("input", { id: "svdk", type: "range", min: 0, max: n, value: 2, step: 1 }, bar);
    const readout = html("span", { class: "readout" }, bar);

    const draw = () => {
      const k = Number(slider.value);
      rebuilt.replaceChildren();
      leftover.replaceChildren();
      const kept = spec.recon[k];
      matrix(rebuilt, kept, xs[1], mapY, cell, 1.4);
      matrix(leftover, kept.map((row, i) => row.map((v, j) => spec.table[i][j] - v)),
             xs[2], mapY, cell, 1.4);
      titleMid.textContent = `REBUILT FROM ${k} LAYER${k === 1 ? "" : "S"}`;
      bars.forEach((b, i) => b.setAttribute("class", i < k ? "barkept" : "bardrop"));
      const share = spec.share.slice(0, k).reduce((a, b) => a + b, 0);
      readout.textContent = `${k} layer${k === 1 ? "" : "s"} hold ${share.toFixed(0)}% of the table`;
    };
    slider.addEventListener("input", draw);
    draw();
    return box;
  }

  // ---- Gradient descent on one pair --------------------------------------

  function player(box, last, onFrame, start = true, speed = 140) {
    // Play/pause + scrubber shared by the animations.
    const bar = controls(box);
    const play = html("button", { type: "button", class: "primary" }, bar, "Pause");
    const again = html("button", { type: "button" }, bar, "Restart");
    const slider = html("input", { type: "range", min: 0, max: last, value: 0, step: 1 }, bar);
    const readout = html("span", { class: "readout" }, bar);
    let timer = null;
    const stop = () => { clearInterval(timer); timer = null; play.textContent = "Play"; };
    const go = () => {
      if (Number(slider.value) >= last) slider.value = 0;
      play.textContent = "Pause";
      clearInterval(timer);
      timer = setInterval(() => {
        if (!box.isConnected) { clearInterval(timer); return; }   // screen was cleared
        const next = Number(slider.value) + 1;
        slider.value = next;
        update();
        if (next >= last) stop();
      }, speed);
    };
    const update = () => { onFrame(Number(slider.value), readout); };
    play.addEventListener("click", () => (timer ? stop() : go()));
    again.addEventListener("click", () => { slider.value = 0; update(); go(); });
    slider.addEventListener("input", () => { stop(); update(); });
    update();
    stop();
    // Start when the picture scrolls into view, not while it is still off-screen.
    if (start && "IntersectionObserver" in window) {
      const watcher = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { watcher.disconnect(); go(); }
      }, { threshold: 0.6 });
      watcher.observe(box);
    } else if (start) {
      go();
    }
    return { slider, update, go, stop };
  }

  function gdPair(spec) {
    const { box, canvas } = frame("Gradient descent on one pair of numbers", 660, 270);
    const frames = spec.frames;
    const x0 = 60, span = 540, max = 1.6;
    const at = (v) => x0 + (v / max) * span;

    const names = spec.names || ["wagged", "tail"];
    text(canvas, 20, 24, `THE GUESS (${names[0]} × ${names[1]}) MOVES TOWARD THE TARGET`, "figtitle");
    svg("rect", { x: x0, y: 44, width: span, height: 22, rx: 4, class: "track" }, canvas);
    const guessBar = svg("rect", { x: x0, y: 44, width: 0, height: 22, rx: 4, class: "barkept" }, canvas);
    svg("line", { x1: at(spec.target), x2: at(spec.target), y1: 36, y2: 74, class: "targetline" }, canvas);
    text(canvas, at(spec.target), 90, `target ${spec.target.toFixed(2)}`, "axis", { "text-anchor": "middle" });
    const guessLabel = text(canvas, x0, 90, "", "axis");

    text(canvas, 20, 132, "THE TWO NUMBERS BEING ADJUSTED", "figtitle");
    const boxes = [[names[0], 90], [names[1], 330]].map(([name, x]) => {
      svg("rect", { x, y: 146, width: 150, height: 44, rx: 8, class: "numbox" }, canvas);
      text(canvas, x + 14, 164, name, "axis");
      return text(canvas, x + 136, 180, "", "bignum", { "text-anchor": "end" });
    });
    text(canvas, 262, 176, "×", "op");
    const errorLabel = text(canvas, 520, 172, "", "axis");

    player(box, frames.length - 1, (r, readout) => {
      const [a, b, guess, error] = frames[r];
      guessBar.setAttribute("width", Math.max(0, at(Math.min(guess, max)) - x0));
      guessLabel.textContent = `guess ${guess.toFixed(3)}`;
      boxes[0].textContent = a.toFixed(3);
      boxes[1].textContent = b.toFixed(3);
      errorLabel.textContent = `error ${error >= 0 ? "+" : ""}${error.toFixed(3)}`;
      readout.textContent = `round ${r}`;
    });
    return box;
  }

  // ---- Gradient descent on the whole table -------------------------------

  function gdAll(spec) {
    const n = spec.words.length;
    const { box, canvas } = frame("Gradient descent moving every word", 700, 360);
    const rounds = spec.runs[0].frames.length - 1;

    let reach = 0.5;
    spec.runs.forEach((run) => run.frames.forEach((f) =>
      f.pos.forEach((p) => p.forEach((v) => { reach = Math.max(reach, Math.abs(v)); }))));
    reach *= 1.12;
    const maxLoss = Math.min(110, Math.max(...spec.runs.flatMap((r) => r.frames.map((f) => f.loss))) * 1.04);
    const minLoss = Math.min(spec.best, ...spec.runs.flatMap((r) => r.frames.map((f) => f.loss))) - 3;

    // scatter
    const sx = 10, sy = 36, side = 310;
    text(canvas, sx, 22, "EVERY WORD AS TWO NUMBERS", "figtitle");
    svg("rect", { x: sx, y: sy, width: side, height: side, rx: 8, class: "plotbox" }, canvas);
    const px = (v) => sx + (v / reach + 1) / 2 * side;
    const py = (v) => sy + (1 - (v / reach + 1) / 2) * side;
    const dots = spec.words.map((word, i) => {
      const colour = spec.pets[i] ? PETS : BAKING;
      const dot = svg("circle", { r: 4, style: `fill:${colour}` }, canvas);
      const label = text(canvas, 0, 0, word, "tiny", { style: `fill:${colour}` });
      return { dot, label };
    });

    // loss chart
    const cx = 370, cy = 36, cw = 310, ch = 310;
    text(canvas, cx, 22, "TOTAL ERROR, ROUND BY ROUND", "figtitle");
    svg("rect", { x: cx, y: cy, width: cw, height: ch, rx: 8, class: "plotbox" }, canvas);
    const lx = (r) => cx + 8 + (r / rounds) * (cw - 16);
    const ly = (v) => cy + 8 + (1 - (v - minLoss) / (maxLoss - minLoss)) * (ch - 16);
    svg("line", { x1: cx + 8, x2: cx + cw - 8, y1: ly(spec.best), y2: ly(spec.best), class: "targetline" }, canvas);
    text(canvas, cx + cw - 12, ly(spec.best) - 6, `SVD's best: ${spec.best}`, "axis", { "text-anchor": "end" });
    const curve = svg("polyline", { fill: "none", class: "curve" }, canvas);
    const now = svg("circle", { r: 4.5, class: "nowdot" }, canvas);
    const lossText = text(canvas, cx + 14, cy + 22, "", "axis");
    text(canvas, cx + cw / 2, cy + ch + 14, "rounds →", "tiny", { "text-anchor": "middle" });

    // controls
    const picker = controls(box);
    html("span", { class: "readout" }, picker, "Step size:");
    let current = spec.runs.findIndex((r) => r.good);
    if (current < 0) current = 0;
    const pick = spec.runs.map((run, i) => {
      const button = html("button", { type: "button" }, picker, run.label || String(run.step));
      button.addEventListener("click", () => { current = i; marks(); handle.slider.value = 0; handle.update(); handle.go(); });
      return button;
    });
    const marks = () => pick.forEach((b, i) => b.classList.toggle("primary", i === current));
    marks();

    const handle = player(box, rounds, (r, readout) => {
      const run = spec.runs[current].frames;
      run[r].pos.forEach(([a, b], i) => {
        dots[i].dot.setAttribute("cx", px(a)); dots[i].dot.setAttribute("cy", py(b));
        dots[i].label.setAttribute("x", px(a) + 6); dots[i].label.setAttribute("y", py(b) + 3);
      });
      curve.setAttribute("points", run.slice(0, r + 1).map((f, k) => `${lx(k)},${ly(f.loss)}`).join(" "));
      now.setAttribute("cx", lx(r)); now.setAttribute("cy", ly(run[r].loss));
      lossText.textContent = `error ${run[r].loss.toFixed(1)}`;
      readout.textContent = `round ${r}`;
    }, true, 90);
    return box;
  }

  // ---- Golf: SVD (one step) next to gradient descent (many small steps) ----

  function golfRace(spec) {
    const { box, canvas } = frame("SVD and gradient descent on the same golf table", 720, 392);
    const size = 42, y0 = 58, xs = [80, 420];
    const frames = spec.frames, last = frames.length - 1;
    const labels = (x) => {
      spec.cols.forEach((c, j) => text(canvas, x + j * size + size / 2, y0 - 6, c, "axs", { "text-anchor": "middle" }));
      spec.rows.forEach((r, i) => text(canvas, x - 8, y0 + i * size + size / 2 + 4, r, "axs", { "text-anchor": "end" }));
    };
    text(canvas, 20, 22, "SVD: ONE STEP, EXACT", "figtitle");
    text(canvas, 360, 22, "GRADIENT DESCENT: SMALL CORRECTIONS", "figtitle");
    labels(xs[0]); labels(xs[1]);
    matrix(canvas, spec.svd.matrix, xs[0], y0, size, 5, { numbers: true, digits: 2 });
    text(canvas, xs[0], y0 + 3 * size + 20, `error ${spec.svd.error.toFixed(2)}`, "axis");
    text(canvas, xs[0], y0 + 3 * size + 38, "best any one-layer answer can do", "tiny");
    const guess = svg("g", {}, canvas);
    const gdError = text(canvas, xs[1], y0 + 3 * size + 20, "", "axis");
    const gdHoles = text(canvas, xs[1] - 60, y0 + 3 * size + 38, "", "tiny");
    const gdPlayers = text(canvas, xs[1] - 60, y0 + 3 * size + 52, "", "tiny");

    const cx = 80, cy = 292, cw = 600, ch = 80;
    text(canvas, 20, cy - 8, "TOTAL ERROR, ROUND BY ROUND (log scale)", "figtitle");
    svg("rect", { x: cx, y: cy, width: cw, height: ch, rx: 6, class: "plotbox" }, canvas);
    const top = Math.log(frames[0].loss), bottom = Math.log(spec.svd.error) - 0.4;
    const lx = (r) => cx + 8 + (r / last) * (cw - 16);
    const ly = (v) => cy + 6 + (1 - (Math.log(v) - bottom) / (top - bottom)) * (ch - 12);
    svg("line", { x1: cx + 8, x2: cx + cw - 8, y1: ly(spec.svd.error), y2: ly(spec.svd.error), class: "targetline" }, canvas);
    text(canvas, cx + cw - 12, ly(spec.svd.error) - 6, "SVD's answer", "tiny", { "text-anchor": "end" });
    const curve = svg("polyline", { fill: "none", class: "curve" }, canvas);
    const now = svg("circle", { r: 4, class: "nowdot" }, canvas);
    text(canvas, cx + cw / 2, cy + ch + 14, "rounds →", "tiny", { "text-anchor": "middle" });

    player(box, last, (r, readout) => {
      const f = frames[r];
      guess.replaceChildren();
      matrix(guess, f.guess, xs[1], y0, size, 5, { numbers: true, digits: 2 });
      gdError.textContent = `error ${f.loss.toFixed(2)}`;
      gdHoles.textContent = `hole numbers:   ${f.p.map((v) => v.toFixed(2)).join("  ")}`;
      gdPlayers.textContent = `player numbers: ${f.q.map((v) => v.toFixed(2)).join("  ")}`;
      curve.setAttribute("points", frames.slice(0, r + 1).map((g, k) => `${lx(k)},${ly(g.loss)}`).join(" "));
      now.setAttribute("cx", lx(r)); now.setAttribute("cy", ly(f.loss));
      readout.textContent = `round ${r}`;
    }, true, 110);
    return box;
  }

  const KINDS = { "svd-mini": svdMini, "svd-real": svdReal, "gd-pair": gdPair, "gd-all": gdAll, "golf-race": golfRace };

  function render(spec) {
    const draw = KINDS[spec.kind];
    return draw ? draw(spec) : document.createComment("unknown figure");
  }

  return { render };
})();
