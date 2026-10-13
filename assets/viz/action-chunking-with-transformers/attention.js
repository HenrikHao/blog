// One action query attending to five memory tokens, in two dimensions so the
// dot product can be read off the picture.
Viz.mount("attention", (root) => {
  const KEY_SCALE = 36; // px per unit in the key plane
  const VALUE_SCALE = 58;
  const QUERY_LENGTH = 2;
  const D_K = 2;

  const rad = (degrees) => (degrees * Math.PI) / 180;
  const polar = (r, degrees) => [r * Math.cos(rad(degrees)), r * Math.sin(rad(degrees))];

  const tokens = [
    { name: "z", key: polar(0.7, 200), value: [-0.95, -0.55], label: [0, 18, "middle"] },
    { name: "joints", key: polar(1.0, 100), value: [-0.6, 0.85], label: [-10, 4, "end"] },
    { name: "gripper", key: polar(1.1, 60), value: [0.55, 0.95], label: [10, 4, "start"] },
    { name: "cube", key: polar(1.2, -10), value: [0.9, 0.05], label: [10, 4, "start"] },
    { name: "table", key: polar(0.9, 300), value: [0.25, -0.95], label: [10, 4, "start"] },
  ];

  const notes = [
    "Five tokens the decoder can read from, and one action query. Each token has a <strong>key</strong> (what it advertises) and a <strong>value</strong> (what it hands over).",
    "<strong>1 · Compare.</strong> Dot the query with every key. A key pointing the same way as the query scores high; one pointing away goes negative.",
    "<strong>2 · Scale.</strong> Divide by √d<sub>k</sub>. Here d<sub>k</sub> = 2; in ACT each of the 8 heads has d<sub>k</sub> = 64. This keeps the softmax out of its flat, saturated region.",
    "<strong>3 · Softmax.</strong> Exponentiate and normalise. Scores become weights that are positive and add up to 1.",
    "<strong>4 · Mix.</strong> The output is the weighted average of the values, so it sits nearest the tokens with the biggest weights. Drag the orange query and watch it move.",
  ];

  const keysSvg = root.querySelector("[data-keys]");
  const valuesSvg = root.querySelector("[data-values]");
  const rowsEl = root.querySelector("[data-rows]");
  const noteEl = root.querySelector("[data-note]");
  const angleInput = root.querySelector("[data-angle]");
  const angleOut = root.querySelector("[data-angle-out]");
  const stepButtons = root.querySelectorAll("[data-step]");
  const playButton = root.querySelector("[data-play]");

  let angle = Number(angleInput.value);
  let step = 0;
  let timers = [];

  // --- key plane -----------------------------------------------------------

  function arrow(group) {
    return { line: Viz.svg("line", { x1: 0, y1: 0 }, group), head: Viz.svg("polygon", {}, group) };
  }

  function placeArrow(parts, x, y) {
    const length = Math.hypot(x, y) || 1;
    const ux = x / length;
    const uy = y / length;
    const bx = x - ux * 8;
    const by = y - uy * 8;
    parts.line.setAttribute("x2", bx);
    parts.line.setAttribute("y2", by);
    parts.head.setAttribute(
      "points",
      `${x},${y} ${bx - uy * 3.5},${by + ux * 3.5} ${bx + uy * 3.5},${by - ux * 3.5}`
    );
  }

  Viz.svg("line", { class: "axis", x1: -96, y1: 0, x2: 96, y2: 0 }, keysSvg);
  Viz.svg("line", { class: "axis", x1: 0, y1: -96, x2: 0, y2: 96 }, keysSvg);

  tokens.forEach((token) => {
    const x = token.key[0] * KEY_SCALE;
    const y = -token.key[1] * KEY_SCALE;
    token.keyGroup = Viz.svg("g", { class: "key" }, keysSvg);
    token.keyArrow = Viz.svg("g", { class: "shaft" }, token.keyGroup);
    placeArrow(arrow(token.keyArrow), x, y);
    const length = Math.hypot(x, y);
    const text = Viz.svg(
      "text",
      {
        x: x + (x / length) * 12,
        y: y + (y / length) * 12 + 4,
        "text-anchor": x > 12 ? "start" : x < -12 ? "end" : "middle",
      },
      token.keyGroup
    );
    text.textContent = token.name;
  });

  const queryGroup = Viz.svg("g", { class: "query" }, keysSvg);
  const queryArrow = arrow(queryGroup);
  const handle = Viz.svg("circle", { class: "handle", r: 6 }, queryGroup);
  const grab = Viz.svg("circle", { class: "grab", r: 18 }, queryGroup);

  // --- value plane ---------------------------------------------------------

  Viz.svg("line", { class: "axis", x1: -96, y1: 0, x2: 96, y2: 0 }, valuesSvg);
  Viz.svg("line", { class: "axis", x1: 0, y1: -96, x2: 0, y2: 96 }, valuesSvg);
  const linkLayer = Viz.svg("g", {}, valuesSvg);

  tokens.forEach((token) => {
    token.vx = token.value[0] * VALUE_SCALE;
    token.vy = -token.value[1] * VALUE_SCALE;
    token.link = Viz.svg("line", { class: "link", x2: token.vx, y2: token.vy }, linkLayer);
    const group = Viz.svg("g", { class: "val" }, valuesSvg);
    token.dot = Viz.svg("circle", { cx: token.vx, cy: token.vy, r: 5 }, group);
    token.valueLabel = Viz.svg(
      "text",
      { x: token.vx + token.label[0], y: token.vy + token.label[1], "text-anchor": token.label[2] },
      group
    );
    token.valueLabel.textContent = token.name;
  });

  const outGroup = Viz.svg("g", { class: "out" }, valuesSvg);
  Viz.svg("circle", { r: 6 }, outGroup);
  const outLabel = Viz.svg("text", { y: -11, "text-anchor": "middle" }, outGroup);
  outLabel.textContent = "output";

  // --- table ---------------------------------------------------------------

  function cell(className, text, shownFrom, index) {
    const el = document.createElement("div");
    el.className = className;
    el.textContent = text;
    if (shownFrom !== undefined) {
      el.classList.add("att-cell");
      el.dataset.from = shownFrom;
      el.style.setProperty("--i", index);
    }
    rowsEl.appendChild(el);
    return el;
  }

  cell("hd", "token");
  cell("hd num", "q · k");
  cell("hd num", "÷ √d");
  cell("hd", "weight");
  tokens.forEach((token, index) => {
    cell("", token.name);
    token.scoreEl = cell("num", "", 1, index);
    token.scaledEl = cell("num", "", 2, index);
    const bar = cell("att-bar", "", 3, index);
    token.barEl = bar;
    bar.appendChild(document.createElement("i"));
    token.pctEl = bar.appendChild(document.createElement("span"));
  });
  const cells = rowsEl.querySelectorAll(".att-cell");

  // --- state -> picture ----------------------------------------------------

  function render() {
    const query = polar(QUERY_LENGTH, angle);
    const scores = tokens.map((token) => token.key[0] * query[0] + token.key[1] * query[1]);
    const scaled = scores.map((score) => score / Math.sqrt(D_K));
    const exps = scaled.map(Math.exp);
    const total = exps.reduce((sum, value) => sum + value, 0);
    const weights = exps.map((value) => value / total);
    const top = Math.max(...weights);

    const qx = query[0] * KEY_SCALE;
    const qy = -query[1] * KEY_SCALE;
    placeArrow(queryArrow, qx, qy);
    [handle, grab].forEach((circle) => {
      circle.setAttribute("cx", qx);
      circle.setAttribute("cy", qy);
    });

    let ox = 0;
    let oy = 0;
    tokens.forEach((token, index) => {
      const weight = weights[index];
      const share = weight / top;
      token.scoreEl.textContent = scores[index].toFixed(2);
      token.scaledEl.textContent = scaled[index].toFixed(2);
      token.barEl.style.setProperty("--w", step >= 3 ? weight : 0);
      token.pctEl.textContent = `${Math.round(weight * 100)}%`;
      token.keyArrow.style.opacity = step >= 3 ? 0.25 + 0.75 * share : 0.8;
      token.dot.setAttribute("r", step >= 4 ? 4 + 6 * Math.sqrt(weight) : 5);
      token.dot.style.opacity = step >= 4 ? 0.35 + 0.65 * share : 0.55;
      ox += weight * token.vx;
      oy += weight * token.vy;
    });
    tokens.forEach((token, index) => {
      token.link.setAttribute("x1", ox);
      token.link.setAttribute("y1", oy);
      token.link.setAttribute("stroke-width", 0.75 + 3.5 * weights[index]);
      token.link.style.opacity = step >= 4 ? 0.2 + 0.7 * (weights[index] / top) : 0;
    });
    outGroup.style.transform = step >= 4 ? `translate(${ox}px, ${oy}px)` : "translate(0px, 0px)";
    outGroup.style.opacity = step >= 4 ? 1 : 0;

    cells.forEach((el) => el.classList.toggle("on", step >= Number(el.dataset.from)));
    stepButtons.forEach((button) =>
      button.setAttribute("aria-pressed", String(Number(button.dataset.step) === step))
    );
    noteEl.innerHTML = notes[step];
    angleOut.textContent = `${Math.round(angle)}°`;
  }

  function setStep(next) {
    step = next;
    render();
  }

  function stopPlaying() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function play() {
    stopPlaying();
    setStep(0);
    [1, 2, 3, 4].forEach((next, index) => {
      timers.push(setTimeout(() => setStep(next), 700 + index * 1900));
    });
    playButton.textContent = "Replay";
  }

  // --- input ---------------------------------------------------------------

  stepButtons.forEach((button) =>
    button.addEventListener("click", () => {
      stopPlaying();
      setStep(Number(button.dataset.step));
    })
  );
  playButton.addEventListener("click", play);

  angleInput.addEventListener("input", () => {
    angle = Number(angleInput.value);
    root.classList.add("is-dragging");
    render();
  });
  angleInput.addEventListener("change", () => root.classList.remove("is-dragging"));

  function dragTo(event) {
    const box = keysSvg.getBoundingClientRect();
    const x = event.clientX - (box.left + box.width / 2);
    const y = event.clientY - (box.top + box.height / 2);
    angle = (Math.atan2(-y, x) * 180) / Math.PI;
    if (angle < 0) angle += 360;
    angleInput.value = Math.round(angle);
    render();
  }

  grab.addEventListener("pointerdown", (event) => {
    grab.setPointerCapture(event.pointerId);
    root.classList.add("is-dragging");
    stopPlaying();
    if (step < 4) step = 4;
    dragTo(event);
  });
  grab.addEventListener("pointermove", (event) => {
    if (grab.hasPointerCapture(event.pointerId)) dragTo(event);
  });
  ["pointerup", "pointercancel"].forEach((type) =>
    grab.addEventListener(type, () => root.classList.remove("is-dragging"))
  );

  // --- start ---------------------------------------------------------------

  if (Viz.reducedMotion) {
    setStep(4);
  } else {
    setStep(0);
    Viz.whenVisible(root, play);
  }
});
