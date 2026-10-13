// Who asks whom: self-attention and cross-attention drawn with the same parts.
// The figure appears twice in the post, as variant "self" and variant "cross".
// Each asking token goes through three phases: it sends its query, the tokens
// being read answer with keys (which gives the weights), and their values flow
// back. The weights are toy numbers chosen to look sensible.
Viz.mount("attention-flow", (root) => {
  const MEMORY = [
    { name: "z", short: "z" },
    { name: "joints", short: "jnt" },
    { name: "gripper patch", short: "grip" },
    { name: "cube patch", short: "cube" },
    { name: "table patch", short: "tbl" },
    { name: "wall patch", short: "wall" },
  ];

  const VARIANTS = {
    self: {
      askers: MEMORY,
      answerers: MEMORY,
      askTitle: "asking",
      readTitle: "being read (the same tokens)",
      mapTitle: "Attention map, 6 × 6",
      first: 2,
      // Scores before the softmax: one row per asking token.
      scores: [
        [1.0, 0.6, 0.6, 0.6, 0.4, 0.4],
        [0.2, 1.4, 2.0, 0.9, 0.0, -0.5],
        [0.2, 1.6, 1.2, 2.1, 0.3, -0.6],
        [0.0, 0.5, 2.0, 1.5, 0.9, -0.5],
        [0.0, -0.3, 0.3, 1.0, 1.6, 0.9],
        [0.6, -0.5, -0.4, -0.2, 0.9, 1.8],
      ],
      who: (asker) => `the ${asker.name}${asker.name.endsWith("patch") ? "" : " token"}`,
      reads: "every token, itself included",
      collected: (asker) => `What the ${asker.name}${asker.name.endsWith("patch") ? "" : " token"} collects, and adds to its own vector`,
    },
    cross: {
      askers: [
        { name: "action now", short: "now" },
        { name: "action +1 s", short: "+1 s" },
        { name: "action +2 s", short: "+2 s" },
        { name: "action +3 s", short: "+3 s" },
      ],
      answerers: MEMORY,
      askTitle: "action queries",
      readTitle: "memory (encoder output)",
      mapTitle: "Attention map, 4 × 6",
      first: 0,
      scores: [
        [-0.6, 0.4, 2.0, 1.1, 0.1, 0.3],
        [-0.6, 0.2, 1.7, 1.6, 0.4, 0.1],
        [-0.5, 0.0, 1.2, 2.0, 0.9, 0.0],
        [-0.4, -0.2, 0.7, 2.0, 1.5, 0.0],
      ],
      who: (asker) => `the query for the ${asker.name}`,
      reads: "every memory token",
      collected: (asker) => `What the query for the ${asker.name} collects, on its way to becoming that action`,
    },
  };

  const variant = VARIANTS[root.dataset.variant] || VARIANTS.self;
  const isSelf = variant === VARIANTS.self;
  const { askers, answerers } = variant;
  const weights = variant.scores.map((row) => {
    const exps = row.map(Math.exp);
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
  });

  const stage = root.querySelector("[data-stage]");
  const matrix = root.querySelector("[data-matrix]");
  const recipe = root.querySelector("[data-recipe]");
  const recipeTitle = root.querySelector("[data-recipe-title]");
  const note = root.querySelector("[data-note]");
  const playButton = root.querySelector("[data-play]");
  const phaseButtons = root.querySelectorAll("[data-phase]");
  root.querySelector("[data-matrix-title]").textContent = variant.mapTitle;

  // --- stage geometry ---------------------------------------------------------
  // On a phone the stage is drawn narrower, so the labels are not scaled down to nothing.
  const narrow = root.clientWidth < 420;
  const W = narrow ? 360 : 460;
  const CHIP = { width: narrow ? 104 : 132, height: 30 };
  const PITCH = 42;
  const TOP = 26;
  const rows = Math.max(askers.length, answerers.length);
  const H = TOP + rows * PITCH;
  const leftX = 0;
  const rightX = W - CHIP.width;
  const yOf = (index, count) => TOP + ((rows - count) / 2 + index) * PITCH + CHIP.height / 2;
  const from = (i) => ({ x: leftX + CHIP.width + 28, y: yOf(i, askers.length) });
  const to = (j) => ({ x: rightX - 28, y: yOf(j, answerers.length) });

  stage.setAttribute("viewBox", `0 0 ${W} ${H}`);

  function text(parent, content, attrs) {
    const el = Viz.svg("text", attrs, parent);
    el.textContent = content;
    return el;
  }

  text(stage, variant.askTitle, { class: "col-title", x: leftX, y: 12 });
  text(stage, variant.readTitle, { class: "col-title", x: W, y: 12, "text-anchor": "end" });

  const linkLayer = Viz.svg("g", {}, stage);
  const guides = answerers.map(() => Viz.svg("line", { class: "guide" }, linkLayer));
  const links = answerers.map(() => Viz.svg("line", { class: "link" }, linkLayer));

  function chip(x, y, label, className) {
    const group = Viz.svg("g", { class: `chip ${className}` }, stage);
    Viz.svg("rect", { x, y: y - CHIP.height / 2, width: CHIP.width, height: CHIP.height, rx: 6 }, group);
    text(group, label, { x: x + CHIP.width / 2, y: y + 4, "text-anchor": "middle" });
    return group;
  }

  function tag(x, y, letter) {
    const group = Viz.svg("g", { class: `tag ${letter}` }, stage);
    Viz.svg("circle", { cx: x, cy: y, r: 9 }, group);
    text(group, letter, { x, y: y + 3.5 });
    return group;
  }

  const askerChips = askers.map((asker, i) => {
    const group = chip(leftX, yOf(i, askers.length), asker.name, "can-ask");
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    group.setAttribute("aria-label", `Let ${variant.who(asker)} ask`);
    return group;
  });
  answerers.forEach((answerer, j) => chip(rightX, yOf(j, answerers.length), answerer.name, ""));
  const qTags = askers.map((_, i) => tag(leftX + CHIP.width + 14, yOf(i, askers.length), "q"));
  const kTags = answerers.map((_, j) => tag(rightX - 14, yOf(j, answerers.length), "k"));
  const vTags = answerers.map((_, j) => tag(rightX - 14, yOf(j, answerers.length), "v"));
  const qPulses = answerers.map(() => Viz.svg("circle", { class: "pulse q", r: 4.5 }, stage));
  const vPulses = answerers.map(() => Viz.svg("circle", { class: "pulse v" }, stage));

  // --- attention map ----------------------------------------------------------
  const CELL = 27;
  const LABEL = 44;
  matrix.setAttribute("width", LABEL + answerers.length * CELL);
  matrix.setAttribute("height", 16 + askers.length * CELL);
  answerers.forEach((answerer, j) =>
    text(matrix, answerer.short, { class: "m-label", x: LABEL + j * CELL + CELL / 2, y: 10, "text-anchor": "middle" })
  );
  const cells = askers.map((asker, i) => {
    text(matrix, asker.short, { class: "m-label", x: LABEL - 6, y: 16 + i * CELL + CELL / 2 + 3, "text-anchor": "end" });
    return answerers.map((answerer, j) => {
      const x = LABEL + j * CELL + 1;
      const y = 16 + i * CELL + 1;
      Viz.svg("rect", { class: "cell-bg", x, y, width: CELL - 2, height: CELL - 2, rx: 2 }, matrix);
      const fill = Viz.svg("rect", { class: "cell", x, y, width: CELL - 2, height: CELL - 2, rx: 2, opacity: 0 }, matrix);
      const title = Viz.svg("title", {}, fill);
      title.textContent = `${asker.name} reading ${answerer.name}: ${Math.round(weights[i][j] * 100)}%`;
      const label = text(matrix, "", { class: "cell-text", x: x + (CELL - 2) / 2, y: y + (CELL - 2) / 2 + 3 });
      return { fill, label };
    });
  });
  const rowMark = Viz.svg("rect", { class: "row-mark", x: LABEL, width: answerers.length * CELL, height: CELL, rx: 3 }, matrix);

  // --- recipe -----------------------------------------------------------------
  const segments = answerers.map(() => recipe.appendChild(document.createElement("i")));

  // --- state ------------------------------------------------------------------
  const state = { asker: variant.first, phase: 0 };
  const seen = new Set();
  let timers = [];
  let playing = false;

  function fly(circle, start, end, radius, delay) {
    circle.setAttribute("r", radius);
    if (Viz.reducedMotion) return;
    circle.animate(
      [
        { transform: `translate(${start.x}px, ${start.y}px)`, opacity: 0 },
        { transform: `translate(${start.x}px, ${start.y}px)`, opacity: 1, offset: 0.12 },
        { transform: `translate(${end.x}px, ${end.y}px)`, opacity: 1, offset: 0.88 },
        { transform: `translate(${end.x}px, ${end.y}px)`, opacity: 0 },
      ],
      { duration: 850, delay, easing: "ease-in-out" }
    );
  }

  function render(animate) {
    const { asker, phase } = state;
    const row = weights[asker];
    const top = Math.max(...row);
    const start = from(asker);
    if (phase >= 2) seen.add(asker);

    askerChips.forEach((group, i) => group.classList.toggle("is-asking", i === asker));
    qTags.forEach((group, i) => group.classList.toggle("on", i === asker && phase >= 1));
    kTags.forEach((group) => group.classList.toggle("on", phase === 2));
    vTags.forEach((group) => group.classList.toggle("on", phase === 3));

    answerers.forEach((_, j) => {
      const end = to(j);
      [guides[j], links[j]].forEach((line) => {
        line.setAttribute("x1", start.x);
        line.setAttribute("y1", start.y);
        line.setAttribute("x2", end.x);
        line.setAttribute("y2", end.y);
      });
      links[j].style.strokeWidth = phase >= 2 ? 1 + 10 * row[j] : 0;
      links[j].style.opacity = phase >= 2 ? 0.25 + 0.75 * (row[j] / top) : 0;
      if (animate && phase === 1) fly(qPulses[j], start, end, 4.5, j * 40);
      if (animate && phase === 3) fly(vPulses[j], end, start, 2.5 + 7 * Math.sqrt(row[j]), j * 40);
    });

    cells.forEach((cellRow, i) =>
      cellRow.forEach((cell, j) => {
        cell.fill.setAttribute("opacity", seen.has(i) ? 0.12 + 0.88 * weights[i][j] ** 0.7 : 0);
        cell.label.textContent = i === asker && phase >= 2 ? Math.round(weights[i][j] * 100) : "";
        cell.label.style.fill = weights[i][j] > 0.3 ? "var(--viz-on-fill)" : "";
      })
    );
    rowMark.setAttribute("y", 16 + asker * CELL);

    const ranked = row
      .map((weight, j) => ({ weight, name: isSelf && j === asker ? "itself" : answerers[j].name }))
      .sort((a, b) => b.weight - a.weight);
    const list = ranked.map((item) => `${Math.round(item.weight * 100)}% ${item.name}`).join(" · ");
    segments.forEach((segment, j) => (segment.style.flexGrow = phase === 3 ? row[j] : 0));
    recipe.style.visibility = phase === 3 ? "visible" : "hidden";
    recipeTitle.textContent = phase === 3 ? `${variant.collected(askers[asker])}: ${list}` : "";

    const who = variant.who(askers[asker]);
    const Who = who.charAt(0).toUpperCase() + who.slice(1);
    note.innerHTML = [
      `${Who} is about to ask. Use the three steps above, or press <em>Play all</em>.`,
      `<strong>1 · Ask.</strong> ${Who} is turned into a <strong>query</strong> by W<sub>Q</sub> and sends it to ${variant.reads}.`,
      `<strong>2 · Match.</strong> Each token answers with its <strong>key</strong>, made by W<sub>K</sub>. Query times key, then a softmax across the row, gives the weights. Thick lines are good matches, and the row appears in the map.`,
      `<strong>3 · Read.</strong> Each token hands over its <strong>value</strong>, made by W<sub>V</sub>, scaled by its weight. ` +
        (isSelf
          ? `The sum is added to ${who}, which now carries a little of everything it read.`
          : `The sum updates ${who}. The memory tokens themselves are left as they were.`),
    ][phase];

    phaseButtons.forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.phase) === phase)));
  }

  function stop() {
    timers.forEach(clearTimeout);
    timers = [];
    playing = false;
    playButton.textContent = "Play all";
  }

  function go(asker, phase, animate = true) {
    state.asker = asker;
    state.phase = phase;
    render(animate);
  }

  // Runs the three phases for one asking token, then calls `next`.
  function runThrough(asker, next) {
    if (Viz.reducedMotion) {
      go(asker, 3, false);
      if (next) timers.push(setTimeout(next, 1200));
      return;
    }
    go(asker, 1);
    timers.push(setTimeout(() => go(asker, 2), 1100));
    timers.push(setTimeout(() => go(asker, 3), 2500));
    if (next) timers.push(setTimeout(next, 4300));
  }

  function playAll() {
    stop();
    playing = true;
    playButton.textContent = "Pause";
    seen.clear();
    const step = (asker) => {
      if (asker >= askers.length) {
        stop();
        return;
      }
      runThrough(asker, () => step(asker + 1));
    };
    step(0);
  }

  playButton.addEventListener("click", () => (playing ? stop() : playAll()));
  phaseButtons.forEach((button) =>
    button.addEventListener("click", () => {
      stop();
      go(state.asker, Number(button.dataset.phase));
    })
  );
  askerChips.forEach((group, i) => {
    const pick = () => {
      stop();
      runThrough(i);
    };
    group.addEventListener("click", pick);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        pick();
      }
    });
  });

  if (Viz.reducedMotion) {
    go(state.asker, 3, false);
  } else {
    go(state.asker, 0, false);
    Viz.whenVisible(root, () => runThrough(state.asker));
  }
});
