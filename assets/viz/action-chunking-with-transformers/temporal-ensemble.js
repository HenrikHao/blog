// Toy 1-D rollout comparing open-loop chunk execution with temporal ensembling.
// Each forward pass predicts the ideal trajectory plus its own random error, so
// consecutive chunks disagree a little, as real ones do.
Viz.mount("temporal-ensemble", (root) => {
  const STEPS = 80;
  const SHIFT_AT = 40;
  const SHIFT = 0.3;
  const MARGIN = { left: 34, right: 12 };
  const CHART = { top: 20, height: 150 };
  const ROW = 2;
  const STACK = { top: CHART.top + CHART.height + 36, height: STEPS * ROW };
  const HEIGHT = STACK.top + STACK.height + 34;
  const Y_MAX = 1.1;

  const random = Viz.rng(11);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const offset = Array.from({ length: STEPS }, () => gauss() * 0.045);
  const drift = Array.from({ length: STEPS }, () => gauss() * 0.05);

  const state = { t: 46, k: 12, m: 0.01, shift: true };

  const base = (t) => 0.4 + 0.2 * Math.sin((2 * Math.PI * t) / 64);
  const ideal = (t) => base(t) + (state.shift && t >= SHIFT_AT ? SHIFT : 0);
  // What the forward pass made at step s says about step t. A pass made before
  // the target moved cannot know about the move.
  const predict = (s, t) =>
    base(t) + (state.shift && s >= SHIFT_AT ? SHIFT : 0) + offset[s] + (drift[s] * (t - s)) / 20;
  const openLoop = (t) => predict(Math.floor(t / state.k) * state.k, t);

  // Passes that predicted step t, oldest first, with normalised weights
  // w_i = exp(-m * i).
  function votes(t) {
    const first = Math.max(0, t - state.k + 1);
    const raw = Array.from({ length: t - first + 1 }, (_, i) => Math.exp(-state.m * i));
    const total = raw.reduce((sum, w) => sum + w, 0);
    return raw.map((w, i) => ({ pass: first + i, weight: w / total }));
  }
  const ensemble = (t) => votes(t).reduce((sum, vote) => sum + vote.weight * predict(vote.pass, t), 0);

  const chartEl = root.querySelector("[data-chart]");
  const weightsEl = root.querySelector("[data-weights]");
  const readout = root.querySelector("[data-readout]");
  const playButton = root.querySelector("[data-play]");
  const inputs = {
    t: root.querySelector("[data-t]"),
    k: root.querySelector("[data-k]"),
    m: root.querySelector("[data-m]"),
    shift: root.querySelector("[data-shift]"),
  };
  const outputs = {
    t: root.querySelector("[data-t-out]"),
    k: root.querySelector("[data-k-out]"),
    m: root.querySelector("[data-m-out]"),
  };

  let width = 0;
  let cursorLayer = null;
  const series = [
    { key: "ideal", at: ideal },
    { key: "open", at: openLoop },
    { key: "ens", at: ensemble },
  ];

  const cell = () => (width - MARGIN.left - MARGIN.right) / STEPS;
  const x = (t) => MARGIN.left + t * cell();
  const y = (value) => CHART.top + CHART.height * (1 - value / Y_MAX);

  function label(parent, text, attrs) {
    const el = Viz.svg("text", attrs, parent);
    el.textContent = text;
    return el;
  }

  // Everything that does not depend on the current step.
  function drawStatic() {
    chartEl.textContent = "";
    if (!width) return;
    const svg = Viz.svg(
      "svg",
      { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img", "aria-label": "Toy rollout: ideal action, open-loop chunk execution and temporal ensemble over 80 timesteps, above a diagram of the overlapping chunks" },
      chartEl
    );

    label(svg, "Action sent to one joint", { class: "te-title", x: MARGIN.left, y: 11 });
    const grid = Viz.svg("g", { class: "grid" }, svg);
    const axis = Viz.svg("g", { class: "ax" }, svg);
    [0, 0.5, 1].forEach((value) => {
      Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: y(value), y2: y(value) }, grid);
      label(axis, value.toFixed(1), { x: MARGIN.left - 6, y: y(value) + 4, "text-anchor": "end" });
    });

    if (state.shift) {
      Viz.svg("line", { class: "event", x1: x(SHIFT_AT), x2: x(SHIFT_AT), y1: CHART.top, y2: STACK.top + STACK.height }, svg);
      label(svg, "target moves", { class: "muted", x: x(SHIFT_AT) + 5, y: CHART.top + 10 });
    }

    series.forEach(({ key, at }) => {
      const points = Array.from({ length: STEPS }, (_, t) => `${x(t + 0.5).toFixed(1)},${y(at(t)).toFixed(1)}`);
      Viz.svg("path", { class: `line ${key}`, d: `M${points.join("L")}` }, svg);
    });

    label(svg, "One row per forward pass, newest at the bottom", { class: "te-title", x: MARGIN.left, y: STACK.top - 8 });
    const stack = Viz.svg("g", {}, svg);
    for (let s = 0; s < STEPS; s++) {
      Viz.svg(
        "rect",
        {
          class: s % state.k === 0 ? "pass open" : "pass",
          x: x(s),
          y: STACK.top + s * ROW,
          width: x(Math.min(s + state.k, STEPS)) - x(s),
          height: ROW - 0.5,
        },
        stack
      );
    }

    const bottom = STACK.top + STACK.height;
    Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: bottom + 4, y2: bottom + 4 }, axis);
    const every = width < 420 ? 20 : 10;
    for (let t = 0; t <= STEPS; t += every) {
      label(axis, String(t), { x: x(t), y: bottom + 18, "text-anchor": "middle" });
    }
    label(axis, "timestep", { x: width - MARGIN.right, y: bottom + 31, "text-anchor": "end" });

    cursorLayer = Viz.svg("g", {}, svg);
    drawCursor();
  }

  // Everything that moves with the current step.
  function drawCursor() {
    if (!cursorLayer) return;
    cursorLayer.textContent = "";
    const t = state.t;
    const current = votes(t);
    const top = Math.max(...current.map((vote) => vote.weight));

    Viz.svg("rect", { class: "now", x: x(t), y: CHART.top, width: cell(), height: STACK.top + STACK.height - CHART.top }, cursorLayer);
    current.forEach((vote) => {
      Viz.svg(
        "rect",
        {
          class: "vote",
          x: x(t),
          y: STACK.top + vote.pass * ROW,
          width: cell(),
          height: ROW,
          opacity: 0.3 + 0.7 * (vote.weight / top),
        },
        cursorLayer
      );
    });
    series.forEach(({ key, at }) => {
      Viz.svg("circle", { class: `dot ${key}`, cx: x(t + 0.5), cy: y(at(t)), r: 4.5 }, cursorLayer);
    });

    drawWeights(current);
    outputs.t.textContent = t;
    inputs.t.value = t;
  }

  function drawWeights(current) {
    const W = 232;
    const H = 86;
    const n = current.length;
    const slot = Math.min(16, (W - 8) / n);
    const top = Math.max(...current.map((vote) => vote.weight));
    weightsEl.textContent = "";
    const svg = Viz.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Weight given to each chunk voting on the current step, oldest first" }, weightsEl);
    label(svg, `Weights on the ${n} vote${n === 1 ? "" : "s"} for step ${state.t}`, { class: "te-title", x: 0, y: 11 });
    const floor = H - 18;
    current.forEach((vote, i) => {
      const h = Math.max(1, (vote.weight / top) * (floor - 22));
      const bx = 4 + i * slot + 1;
      const bw = slot - 2;
      // Rounded at the data end, square at the baseline.
      Viz.svg(
        "path",
        {
          class: "bar",
          d: `M${bx},${floor} V${floor - h + 2} Q${bx},${floor - h} ${bx + 2},${floor - h} H${bx + bw - 2} Q${bx + bw},${floor - h} ${bx + bw},${floor - h + 2} V${floor} Z`,
        },
        svg
      );
    });
    const axis = Viz.svg("g", { class: "ax" }, svg);
    Viz.svg("line", { x1: 0, x2: W, y1: floor, y2: floor }, axis);
    label(axis, "oldest", { x: 4, y: H - 4 });
    if (n > 3) label(axis, "newest", { x: 4 + n * slot, y: H - 4, "text-anchor": "end" });

    const oldest = current[0];
    const newest = current[n - 1];
    const lean =
      Math.abs(state.m) < 0.005
        ? "Every vote counts the same."
        : state.m > 0
          ? "Older predictions count for more."
          : "Newer predictions count for more.";
    readout.innerHTML =
      `<b>Step ${state.t}:</b> ${n} chunk${n === 1 ? " has" : "s have"} an opinion about this step. ` +
      (n > 1
        ? `The oldest, made at step ${oldest.pass}, gets <b>${(oldest.weight * 100).toFixed(1)}%</b>; the newest, made just now, gets <b>${(newest.weight * 100).toFixed(1)}%</b>. ${lean} `
        : "") +
      `Ideal <b>${ideal(state.t).toFixed(2)}</b>, open loop <b>${openLoop(state.t).toFixed(2)}</b>, ensemble <b>${ensemble(state.t).toFixed(2)}</b>.`;
  }

  // --- playback --------------------------------------------------------------

  const player = Viz.ticker(110, () => {
    if (state.t >= STEPS - 1) {
      stop();
      return;
    }
    state.t += 1;
    drawCursor();
  });

  function stop() {
    player.stop();
    playButton.textContent = "Play";
  }

  playButton.addEventListener("click", () => {
    if (player.running) {
      stop();
      return;
    }
    if (state.t >= STEPS - 1) state.t = 0;
    player.start();
    playButton.textContent = "Pause";
  });

  // --- input -----------------------------------------------------------------

  inputs.t.addEventListener("input", () => {
    stop();
    state.t = Number(inputs.t.value);
    drawCursor();
  });
  inputs.k.addEventListener("input", () => {
    state.k = Number(inputs.k.value);
    outputs.k.textContent = state.k;
    drawStatic();
  });
  inputs.m.addEventListener("input", () => {
    state.m = Number(inputs.m.value);
    outputs.m.textContent = state.m.toFixed(2);
    drawStatic();
  });
  inputs.shift.addEventListener("change", () => {
    state.shift = inputs.shift.checked;
    drawStatic();
  });

  function scrub(event) {
    const box = chartEl.getBoundingClientRect();
    state.t = Viz.clamp(Math.floor((event.clientX - box.left - MARGIN.left) / cell()), 0, STEPS - 1);
    drawCursor();
  }
  chartEl.addEventListener("pointerdown", (event) => {
    stop();
    chartEl.setPointerCapture(event.pointerId);
    scrub(event);
  });
  chartEl.addEventListener("pointermove", (event) => {
    if (chartEl.hasPointerCapture(event.pointerId)) scrub(event);
    else if (event.pointerType === "mouse" && !player.running) scrub(event);
  });

  Viz.onResize(chartEl, (next) => {
    width = next;
    drawStatic();
  });
});
