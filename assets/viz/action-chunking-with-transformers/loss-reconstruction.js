// The reconstruction term on real training samples: demonstrated chunk against
// predicted chunk for one joint, and the absolute error whose average is the L1
// loss. Data from act-attention/scripts/export_loss_figure.py, in the normalised
// units the loss is computed in.
Viz.mount("loss-reconstruction", (root) => {
  const urls = Viz.assets(root);
  const chartEl = root.querySelector("[data-chart]");
  const readout = root.querySelector("[data-readout]");
  const sampleInput = root.querySelector("[data-sample]");
  const sampleOut = root.querySelector("[data-sample-out]");
  const jointSelect = root.querySelector("[data-joint]");

  const MARGIN = { left: 40, right: 12 };
  const TOP = { y: 20, height: 150 };
  const BOTTOM = { y: TOP.y + TOP.height + 40, height: 64 };
  const HEIGHT = BOTTOM.y + BOTTOM.height + 36;

  let meta = null;
  let demo = null;
  let pred = null;
  let width = 0;
  const state = { sample: Number(sampleInput.value), joint: 1 };

  const decode = (text) => new Int16Array(Uint8Array.from(atob(text), (char) => char.charCodeAt(0)).buffer);

  function label(parent, text, attrs) {
    const el = Viz.svg("text", attrs, parent);
    el.textContent = text;
    return el;
  }

  // Target and prediction for one sample and joint, plus which steps are padding.
  function chunk(sample, joint) {
    const start = meta.frames[sample];
    const joints = meta.joints.length;
    const target = [];
    const predicted = [];
    const padded = [];
    for (let i = 0; i < meta.chunk; i++) {
      const step = Math.min(start + i, meta.steps - 1); // padding repeats the last action
      target.push(demo[step * joints + joint] / meta.scale);
      predicted.push(pred[(sample * meta.chunk + i) * joints + joint] / meta.scale);
      padded.push(start + i >= meta.steps);
    }
    return { start, target, predicted, padded };
  }

  function meanError({ target, predicted, padded }) {
    let sum = 0;
    let count = 0;
    target.forEach((value, i) => {
      if (!padded[i]) {
        sum += Math.abs(value - predicted[i]);
        count += 1;
      }
    });
    return { mean: sum / count, count };
  }

  function draw() {
    chartEl.textContent = "";
    if (!meta || !width) return;
    const current = chunk(state.sample, state.joint);
    const { target, predicted, padded } = current;
    const errors = target.map((value, i) => Math.abs(value - predicted[i]));
    const valid = padded.filter((isPad) => !isPad).length;

    const all = target.concat(predicted);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const pad = Math.max(0.08, (hi - lo) * 0.12);
    lo -= pad;
    hi += pad;
    const errMax = Math.max(0.05, ...errors.filter((_, i) => !padded[i])) * 1.15;

    const cell = (width - MARGIN.left - MARGIN.right) / meta.chunk;
    const x = (i) => MARGIN.left + i * cell;
    const yTop = (value) => TOP.y + TOP.height * (1 - (value - lo) / (hi - lo));
    const yErr = (value) => BOTTOM.y + BOTTOM.height * (1 - Math.min(value, errMax) / errMax);

    const svg = Viz.svg(
      "svg",
      { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img", "aria-label": "Demonstrated and predicted action chunk for one joint, with the absolute error at each step underneath" },
      chartEl
    );
    const grid = Viz.svg("g", { class: "grid" }, svg);
    const axis = Viz.svg("g", { class: "ax" }, svg);

    // Padding: steps past the end of the episode.
    if (valid < meta.chunk) {
      [TOP, BOTTOM].forEach((panel) =>
        Viz.svg("rect", { class: "pad", x: x(valid), y: panel.y, width: x(meta.chunk) - x(valid), height: panel.height }, svg)
      );
      const room = x(meta.chunk) - x(valid);
      label(svg, room > 150 ? "past the end of the episode: masked" : room > 54 ? "masked" : "", {
        class: "muted",
        x: x(valid) + 5,
        y: TOP.y + 13,
      });
    }

    const narrow = width < 480; // phones: keep the panel titles inside the figure
    label(svg, narrow ? `${meta.joints[state.joint]} target, in standard deviations` : `Target for ${meta.joints[state.joint]}, in standard deviations from its mean`, { class: "lr-title", x: MARGIN.left, y: 11 });
    const ticks = [lo + (hi - lo) * 0.1, (lo + hi) / 2, hi - (hi - lo) * 0.1];
    ticks.forEach((value) => {
      Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: yTop(value), y2: yTop(value) }, grid);
      label(axis, value.toFixed(2), { x: MARGIN.left - 6, y: yTop(value) + 4, "text-anchor": "end" });
    });
    const path = (values) => `M${values.map((value, i) => `${x(i + 0.5).toFixed(1)},${yTop(value).toFixed(1)}`).join("L")}`;
    Viz.svg("path", { class: "line demo", d: path(target) }, svg);
    Viz.svg("path", { class: "line pred", d: path(predicted) }, svg);

    label(svg, narrow ? "Absolute error at each step" : "Absolute error at each step; its average over the unmasked steps is this joint's L1", { class: "lr-title", x: MARGIN.left, y: BOTTOM.y - 8 });
    [0, errMax / 1.15].forEach((value) => {
      Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: yErr(value), y2: yErr(value) }, grid);
      label(axis, value.toFixed(2), { x: MARGIN.left - 6, y: yErr(value) + 4, "text-anchor": "end" });
    });
    errors.forEach((value, i) => {
      if (padded[i]) return;
      const top = yErr(value);
      Viz.svg("rect", { class: "gap", x: x(i) + 0.5, y: top, width: Math.max(1, cell - 1), height: BOTTOM.y + BOTTOM.height - top }, svg);
    });

    const floor = BOTTOM.y + BOTTOM.height;
    Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: floor, y2: floor }, axis);
    [0, 25, 50, 75, 100].forEach((i) => {
      label(axis, i === 0 ? "now" : `+${(i / meta.fps).toFixed(1)} s`, { x: x(i), y: floor + 16, "text-anchor": i === 0 ? "start" : i === 100 ? "end" : "middle" });
    });
    label(axis, "position in the chunk", { x: width - MARGIN.right, y: floor + 31, "text-anchor": "end" });

    // The loss for this sample: this joint, and all joints as the code computes it.
    const thisJoint = meanError(current);
    let total = 0;
    meta.joints.forEach((_, joint) => {
      total += meanError(chunk(state.sample, joint)).mean;
    });
    sampleOut.textContent = `step ${current.start}`;
    readout.innerHTML =
      `<b>Sample at step ${current.start}:</b> ${valid} of ${meta.chunk} steps count` +
      (valid < meta.chunk ? `, ${meta.chunk - valid} are padding. ` : ". ") +
      `L1 for ${meta.joints[state.joint]} is <b>${thisJoint.mean.toFixed(3)}</b>. ` +
      `Averaged over all ${meta.joints.length} joints, this sample's reconstruction loss is <b>${(total / meta.joints.length).toFixed(3)}</b>.`;
  }

  Viz.onResize(chartEl, (next) => {
    width = next;
    draw();
  });

  Viz.whenNear(root, async () => {
    try {
      meta = await fetch(urls.json).then((response) => response.json());
      demo = decode(meta.demo);
      pred = decode(meta.pred);
      meta.joints.forEach((name, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = name;
        jointSelect.appendChild(option);
      });
      jointSelect.value = state.joint;
      sampleInput.max = meta.frames.length - 1;
      sampleInput.disabled = false;
      jointSelect.disabled = false;
      sampleInput.addEventListener("input", () => {
        state.sample = Number(sampleInput.value);
        draw();
      });
      jointSelect.addEventListener("change", () => {
        state.joint = Number(jointSelect.value);
        draw();
      });
      draw();
    } catch (error) {
      console.error("loss-reconstruction: could not load the recording", error);
      readout.textContent = "The recording for this figure could not be loaded.";
    }
  });
});
