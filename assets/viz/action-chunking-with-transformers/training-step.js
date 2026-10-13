// One training sample through a toy encoder-decoder with attention: forward as
// matrix products, the loss, the backward pass and a weight update. The numbers
// are computed here, not drawn: the gradients below were checked against finite
// differences.
Viz.mount("training-step", (root) => {
  // <math>
  // A tiny single-head, single-layer encoder-decoder with attention, small enough
  // to print. Every quantity is a plain array of rows.
  const D = 3;
  const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((sum, a, k) => sum + a * B[k][j], 0)));
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  const add = (A, B) => A.map((row, i) => row.map((a, j) => a + B[i][j]));
  const scale = (A, s) => A.map((row) => row.map((a) => a * s));
  const softmaxRows = (S) =>
    S.map((row) => {
      const top = Math.max(...row);
      const exps = row.map((s) => Math.exp(s - top));
      const total = exps.reduce((sum, e) => sum + e, 0);
      return exps.map((e) => e / total);
    });
  // Gradient through a row-wise softmax: dS = A * (dA - sum(A * dA)).
  const softmaxBack = (A, gA) =>
    A.map((row, i) => {
      const dot = row.reduce((sum, a, j) => sum + a * gA[i][j], 0);
      return row.map((a, j) => a * (gA[i][j] - dot));
    });

  function forward(p, X, target) {
    const c = {};
    c.Q = matmul(X, p.WQ);
    c.K = matmul(X, p.WK);
    c.V = matmul(X, p.WV);
    c.S = scale(matmul(c.Q, transpose(c.K)), 1 / Math.sqrt(D));
    c.A = softmaxRows(c.S);
    c.M = add(X, matmul(c.A, c.V));
    c.Qd = matmul(p.E, p.UQ);
    c.Kd = matmul(c.M, p.UK);
    c.Vd = matmul(c.M, p.UV);
    c.Sd = scale(matmul(c.Qd, transpose(c.Kd)), 1 / Math.sqrt(D));
    c.Ad = softmaxRows(c.Sd);
    c.O = matmul(c.Ad, c.Vd);
    c.pred = matmul(c.O, p.w).map((row) => [row[0] + p.b[0][0]]);
    c.err = c.pred.map((row, i) => [row[0] - target[i][0]]);
    c.loss = c.err.reduce((sum, row) => sum + Math.abs(row[0]), 0) / c.err.length;
    return c;
  }

  function backward(p, X, c) {
    const g = {};
    const r = 1 / Math.sqrt(D);
    g.pred = c.err.map((row) => [Math.sign(row[0]) / c.err.length]);
    g.O = matmul(g.pred, transpose(p.w));
    g.w = matmul(transpose(c.O), g.pred);
    g.b = [[g.pred.reduce((sum, row) => sum + row[0], 0)]];
    g.Ad = matmul(g.O, transpose(c.Vd));
    g.Vd = matmul(transpose(c.Ad), g.O);
    g.Sd = softmaxBack(c.Ad, g.Ad);
    g.Qd = scale(matmul(g.Sd, c.Kd), r);
    g.Kd = scale(matmul(transpose(g.Sd), c.Qd), r);
    g.E = matmul(g.Qd, transpose(p.UQ));
    g.UQ = matmul(transpose(p.E), g.Qd);
    g.UK = matmul(transpose(c.M), g.Kd);
    g.UV = matmul(transpose(c.M), g.Vd);
    g.M = add(matmul(g.Kd, transpose(p.UK)), matmul(g.Vd, transpose(p.UV)));
    g.A = matmul(g.M, transpose(c.V));
    g.V = matmul(transpose(c.A), g.M);
    g.S = softmaxBack(c.A, g.A);
    g.Q = scale(matmul(g.S, c.K), r);
    g.K = scale(matmul(transpose(g.S), c.Q), r);
    g.WQ = matmul(transpose(X), g.Q);
    g.WK = matmul(transpose(X), g.K);
    g.WV = matmul(transpose(X), g.V);
    g.X = add(add(g.M, matmul(g.Q, transpose(p.WQ))), add(matmul(g.K, transpose(p.WK)), matmul(g.V, transpose(p.WV))));
    return g;
  }

  const PARAMS = ["WQ", "WK", "WV", "E", "UQ", "UK", "UV", "w", "b"];
  function step(p, g, rate) {
    const next = {};
    PARAMS.forEach((name) => (next[name] = add(p[name], scale(g[name], -rate))));
    return next;
  }

  function initialParams(random) {
    const matrix = (rows, cols, span) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => (random() * 2 - 1) * span));
    return {
      WQ: matrix(D, D, 1.7), WK: matrix(D, D, 1.7), WV: matrix(D, D, 0.9),
      E: matrix(3, D, 1.0),
      UQ: matrix(D, D, 1.7), UK: matrix(D, D, 1.7), UV: matrix(D, D, 0.9),
      w: matrix(D, 1, 0.9), b: [[0]],
    };
  }

  // One training sample. X is what the backbone and the input projections would
  // hand to the encoder; the target is the demonstrated chunk for one joint.
  const SAMPLE = {
    X: [
      [0.2, -0.1, 0.1],
      [0.9, 0.3, -0.4],
      [0.5, -0.7, 0.8],
      [-0.6, 0.9, 0.4],
    ],
    target: [[0.3], [0.45], [0.55]],
  };
  // </math>

  const RATE = 0.08;
  const TOKENS = ["z", "jnt", "grip", "cube"];
  const QUERIES = ["now", "+1", "+2"];
  const VALUE_SCALE = 1.5;

  const strip = root.querySelector("[data-strip]");
  const titleEl = root.querySelector("[data-title]");
  const detail = root.querySelector("[data-detail]");
  const noteEl = root.querySelector("[data-note]");
  const progress = root.querySelector("[data-progress]");
  const historyEl = root.querySelector("[data-history]");
  const scaleNote = root.querySelector("[data-scale-note]");
  const dirLabel = root.querySelector("[data-dir]");
  const prevButton = root.querySelector("[data-prev]");
  const nextButton = root.querySelector("[data-next]");
  const playButton = root.querySelector("[data-play]");

  // Wires between the blocks on the main row, with an arrowhead for each direction.
  const wires = root.querySelector("[data-wires]");
  [[66, 80], [120, 134], [314, 328], [362, 376], [556, 570], [612, 626]].forEach(([x1, x2]) => {
    Viz.svg("line", { class: "wire", x1, x2, y1: 82, y2: 82 }, wires);
    Viz.svg("polygon", { class: "tip f", points: `${x2},82 ${x2 - 6},78.5 ${x2 - 6},85.5` }, wires);
    Viz.svg("polygon", { class: "tip b", points: `${x1},82 ${x1 + 6},78.5 ${x1 + 6},85.5` }, wires);
  });

  let params = initialParams(Viz.rng(1));
  let cache = null;
  let grads = null;
  let steps = [];
  let index = 0;
  const history = [];

  // --- matrices on screen -----------------------------------------------------

  const mat = (label, data, options = {}) => ({ label, data, kind: "value", ...options });
  const grad = (label, data, options = {}) => ({ label, data, kind: "grad", ...options });
  const product = (left, right, result, extra) => ({
    items: [left, "×", right, extra, "=", result].filter(Boolean),
    link: [left, right, result],
  });

  function drawMatrix(m) {
    const rows = m.data.length;
    const cols = m.data[0].length;
    const top = m.kind === "grad" ? Math.max(1e-9, ...m.data.flat().map(Math.abs)) : VALUE_SCALE;
    const digits = m.kind === "grad" ? 3 : 2;

    const box = document.createElement("div");
    box.className = `mx${m.kind === "grad" ? " is-grad" : ""}`;
    const name = document.createElement("div");
    name.className = "mx-name";
    name.innerHTML = `<b>${m.label}</b> ${rows} × ${cols}${m.caption ? ` · ${m.caption}` : ""}`;
    const grid = document.createElement("div");
    grid.className = "mx-grid";
    grid.style.gridTemplateColumns = `${m.rows ? "auto " : ""}repeat(${cols}, 40px)`;

    const label = (content, extra) => {
      const span = document.createElement("span");
      span.className = `mx-lab ${extra}`;
      span.textContent = content;
      grid.appendChild(span);
    };
    if (m.cols) {
      if (m.rows) label("", "row");
      m.cols.forEach((content) => label(content, "col"));
    }
    const cells = m.data.map((row, i) => {
      if (m.rows) label(m.rows[i], "row");
      return row.map((value) => {
        const span = document.createElement("span");
        span.className = "mx-cell";
        span.textContent = value.toFixed(digits);
        const strength = m.kind === "weight" ? value : Math.min(1, Math.abs(value) / top);
        const colour = value < 0 ? "var(--viz-neg)" : "var(--viz-s1)";
        span.style.background = `color-mix(in srgb, ${colour} ${Math.round(strength * 60)}%, var(--viz-surface-2))`;
        grid.appendChild(span);
        return span;
      });
    });
    box.append(name, grid);
    m.cells = cells;
    return box;
  }

  function drawEquation(equation) {
    const line = document.createElement("div");
    line.className = `ts-eq${equation.wrap ? " wrap" : ""}`;
    equation.items.forEach((item) => {
      if (typeof item === "string") {
        const op = document.createElement("span");
        op.className = "ts-op";
        op.innerHTML = item;
        line.appendChild(op);
      } else if (item.big) {
        const big = document.createElement("span");
        big.className = "ts-big";
        big.innerHTML = item.big;
        line.appendChild(big);
      } else {
        line.appendChild(drawMatrix(item));
      }
    });
    // In a product, cell (i, j) of the result is row i of the left factor times
    // column j of the right factor. Show that on hover.
    if (equation.link) {
      const [left, right, result] = equation.link;
      const marked = () => line.querySelectorAll(".hl").forEach((cell) => cell.classList.remove("hl"));
      result.cells.forEach((row, i) =>
        row.forEach((cell, j) => {
          cell.classList.add("pick");
          cell.addEventListener("pointerenter", () => {
            marked();
            cell.classList.add("hl");
            left.cells[i].forEach((other) => other.classList.add("hl"));
            right.cells.forEach((rightRow) => rightRow[j].classList.add("hl"));
          });
        })
      );
      line.addEventListener("pointerleave", marked);
    }
    return line;
  }

  // --- the steps, rebuilt whenever the weights change --------------------------

  function buildSteps() {
    const p = params;
    const c = cache;
    const g = grads;
    const X = mat("X", SAMPLE.X, { rows: TOKENS });
    const a = mat("a", SAMPLE.target, { rows: QUERIES });
    const tokens = { rows: TOKENS };
    const queries = { rows: QUERIES };
    const map = { rows: TOKENS, cols: TOKENS };
    const cross = { rows: QUERIES, cols: TOKENS };
    const sqrt = "÷ √3";
    const updated = step(p, g, RATE);
    const lossAfter = forward(updated, SAMPLE.X, SAMPLE.target).loss;

    return [
      {
        on: ["sample", "X", "E"],
        title: "A training sample",
        equations: [
          {
            wrap: true,
            items: [
              mat("X", SAMPLE.X, { rows: TOKENS, caption: "the observation, as tokens" }),
              mat("E", p.E, { rows: QUERIES, caption: "action queries, learned" }),
              mat("a", SAMPLE.target, { rows: QUERIES, caption: "chunk to imitate" }),
            ],
          },
        ],
        note: "One training sample is an observation and the chunk the demonstrator executed next. Here the backbone and the input projections have already turned the observation into <strong>X</strong>: four tokens of three numbers each, where ACT has 1202 tokens of 512. <strong>E</strong> holds three action queries, and <strong>a</strong> is the chunk to imitate, for a single joint.",
      },
      {
        on: ["enc-project"],
        title: "Encoder · project: every token gets a query, a key and a value",
        equations: [
          product(X, mat("W<sub>Q</sub>", p.WQ), mat("Q", c.Q, tokens)),
          product(mat("X", SAMPLE.X, tokens), mat("W<sub>K</sub>", p.WK), mat("K", c.K, tokens)),
          product(mat("X", SAMPLE.X, tokens), mat("W<sub>V</sub>", p.WV), mat("V", c.V, tokens)),
        ],
        note: "<strong>Project.</strong> Three matrix products turn every token into its query, key and value. Each product handles all four tokens at once: row <em>i</em> of Q is token <em>i</em> times W<sub>Q</sub>. Hover a cell of a result to see the row and the column it was made from.",
      },
      {
        on: ["enc-compare"],
        title: "Encoder · compare: every query against every key",
        equations: [product(mat("Q", c.Q, tokens), mat("K<sup>T</sup>", transpose(c.K), { cols: TOKENS }), mat("S", c.S, map), sqrt)],
        note: "<strong>Compare.</strong> One more product scores every query against every key. Cell (<em>i</em>, <em>j</em>) of S is the dot product of query <em>i</em> with key <em>j</em>, the same comparison as the arrows in the first figure, and all sixteen are computed together. Nothing here goes token by token, which is why a GPU can do it in one shot.",
      },
      {
        on: ["enc-softmax"],
        title: "Encoder · softmax: scores become weights",
        equations: [{ items: [mat("S", c.S, map), "softmax along each row →", mat("A", c.A, { ...map, kind: "weight" })] }],
        note: "<strong>Softmax.</strong> Each row of S becomes a row of weights that add up to 1. A is the attention map: row <em>i</em> says where token <em>i</em> reads from.",
      },
      {
        on: ["enc-mix", "M"],
        title: "Encoder · mix: blend the values, add them to the tokens",
        equations: [
          (() => {
            const A = mat("A", c.A, { ...map, kind: "weight" });
            const V = mat("V", c.V);
            const M = mat("M", c.M, tokens);
            return { items: [mat("X", SAMPLE.X), "+", A, "×", V, "=", M], link: [A, V, M] };
          })(),
        ],
        note: "<strong>Mix.</strong> A × V blends the values, one blend per row, and the result is added to X. M is the memory that the decoder will read. The real layer also normalises here and applies a small feed-forward network.",
      },
      {
        on: ["dec-project", "E", "M"],
        title: "Decoder · project: queries from E, keys and values from M",
        equations: [
          product(mat("E", p.E, queries), mat("U<sub>Q</sub>", p.UQ), mat("Q′", c.Qd, queries)),
          product(mat("M", c.M, tokens), mat("U<sub>K</sub>", p.UK), mat("K′", c.Kd, tokens)),
          product(mat("M", c.M, tokens), mat("U<sub>V</sub>", p.UV), mat("V′", c.Vd, tokens)),
        ],
        note: "<strong>Decoder, project.</strong> The same three kinds of matrix, with different inputs. The queries come from the action queries E. The keys and the values come from the memory M.",
      },
      {
        on: ["dec-compare", "dec-softmax"],
        title: "Decoder · compare and softmax: each action query against the memory",
        equations: [
          product(mat("Q′", c.Qd, queries), mat("K′<sup>T</sup>", transpose(c.Kd), { cols: TOKENS }), mat("S′", c.Sd, cross), sqrt),
          { items: [mat("S′", c.Sd, cross), "softmax along each row →", mat("A′", c.Ad, { ...cross, kind: "weight" })] },
        ],
        note: "<strong>Decoder, compare and softmax.</strong> S′ and A′ have one row per action query and one column per memory token, so they are 3 × 4 and not square.",
      },
      {
        on: ["dec-mix", "head"],
        title: "Decoder · mix, then the head: one number per action",
        equations: [
          product(mat("A′", c.Ad, { ...cross, kind: "weight" }), mat("V′", c.Vd, tokens), mat("O", c.O, queries)),
          (() => {
            const O = mat("O", c.O, queries);
            const w = mat("w", p.w);
            const pred = mat("â", c.pred, queries);
            return { items: [O, "×", w, `+ b`, "=", pred], link: [O, w, pred] };
          })(),
        ],
        note: "<strong>Decoder, mix, then the head.</strong> Each action query collects its own blend of the memory's values. A linear head then turns each row of O into one number, the predicted action for that step.",
      },
      {
        on: ["loss"],
        title: "Loss: how far is the prediction from the demonstration?",
        equations: [
          {
            items: [
              mat("â", c.pred, queries),
              "−",
              a,
              "=",
              mat("â − a", c.err, queries),
              { big: `L = mean |â − a| = ${c.loss.toFixed(3)}` },
            ],
          },
        ],
        note: "<strong>Loss.</strong> The prediction is compared with the demonstrated chunk. This toy keeps only the L1 term. It leaves out the CVAE encoder and the KL term, and also the ResNet, the extra heads, the normalisation and the feed-forward layers.",
      },
      {
        back: true,
        on: ["loss", "head"],
        title: "Backward · from the loss into the head",
        equations: [
          (() => {
            const gPred = grad("∂L/∂â", g.pred, { ...queries, caption: "sign(â − a) ÷ 3" });
            return product(gPred, mat("w<sup>T</sup>", transpose(p.w)), grad("∂L/∂O", g.O, queries));
          })(),
          product(mat("O<sup>T</sup>", transpose(c.O), { cols: QUERIES }), grad("∂L/∂â", g.pred, queries), grad("∂L/∂w", g.w)),
        ],
        note: "<strong>The backward pass begins.</strong> Backpropagation asks of every number computed on the way forward: if this had been slightly larger, how would the loss have changed? It starts at the loss and reuses the forward matrices in reverse order. With L1 the first gradient is just the sign of each error, divided by the number of steps.",
      },
      {
        back: true,
        on: ["dec-mix"],
        title: "Backward · through the mix: to the weights, and to the values",
        equations: [
          product(grad("∂L/∂O", g.O, queries), mat("V′<sup>T</sup>", transpose(c.Vd), { cols: TOKENS }), grad("∂L/∂A′", g.Ad, cross)),
          product(mat("A′<sup>T</sup>", transpose(c.Ad), { rows: TOKENS, cols: QUERIES, kind: "weight" }), grad("∂L/∂O", g.O, queries), grad("∂L/∂V′", g.Vd, tokens)),
        ],
        note: "<strong>Back through the mix.</strong> O was A′ × V′, so its gradient splits two ways: towards the weights, through V′, and towards the values, through A′. A token that received a large weight on the way forward receives a large share of the gradient on the way back.",
      },
      {
        back: true,
        on: ["dec-softmax", "dec-compare"],
        title: "Backward · through the softmax and the comparison: where to look",
        equations: [
          { items: [grad("∂L/∂A′", g.Ad, cross), "back through the softmax, row by row →", grad("∂L/∂S′", g.Sd, cross)] },
          product(grad("∂L/∂S′", g.Sd, cross), mat("K′", c.Kd, tokens), grad("∂L/∂Q′", g.Qd, queries), sqrt),
          product(grad("∂L/∂S′<sup>T</sup>", transpose(g.Sd), { rows: TOKENS, cols: QUERIES }), mat("Q′", c.Qd, queries), grad("∂L/∂K′", g.Kd, tokens), sqrt),
        ],
        note: "<strong>Back through the softmax and the comparison.</strong> This is the path that teaches the model <em>where to look</em>. The gradient on the weights passes through the softmax to the scores, and from the scores to the queries and the keys.",
      },
      {
        back: true,
        on: ["dec-project", "E", "M"],
        title: "Backward · into the decoder's weights, and on to the memory",
        equations: [
          {
            wrap: true,
            items: [
              grad("∂L/∂U<sub>Q</sub>", g.UQ, { caption: "E<sup>T</sup> × ∂L/∂Q′" }),
              grad("∂L/∂U<sub>K</sub>", g.UK, { caption: "M<sup>T</sup> × ∂L/∂K′" }),
              grad("∂L/∂U<sub>V</sub>", g.UV, { caption: "M<sup>T</sup> × ∂L/∂V′" }),
              grad("∂L/∂E", g.E, { ...queries, caption: "∂L/∂Q′ × U<sub>Q</sub><sup>T</sup>" }),
              grad("∂L/∂M", g.M, { ...tokens, caption: "from K′ and from V′" }),
            ],
          },
        ],
        note: "<strong>Into the decoder's weights, and on to the memory.</strong> Every learned matrix now has its gradient: U<sub>Q</sub> and U<sub>K</sub> from the where-to-look path, U<sub>V</sub> from the what-to-pass-on path, and the action queries E themselves. ∂L/∂M is everything the encoder gets to learn from.",
      },
      {
        back: true,
        on: ["enc-mix", "enc-softmax", "enc-compare", "enc-project", "X"],
        title: "Backward · through the encoder",
        equations: [
          {
            wrap: true,
            items: [
              grad("∂L/∂A", g.A, { ...map, caption: "∂L/∂M × V<sup>T</sup>" }),
              grad("∂L/∂S", g.S, { ...map, caption: "softmax, backwards" }),
              grad("∂L/∂W<sub>Q</sub>", g.WQ, { caption: "X<sup>T</sup> × ∂L/∂Q" }),
              grad("∂L/∂W<sub>K</sub>", g.WK, { caption: "X<sup>T</sup> × ∂L/∂K" }),
              grad("∂L/∂W<sub>V</sub>", g.WV, { caption: "X<sup>T</sup> × ∂L/∂V" }),
              grad("∂L/∂X", g.X, { ...tokens, caption: "continues upstream" }),
            ],
          },
        ],
        note: "<strong>Back through the encoder.</strong> The same four moves in reverse give gradients for W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>. In the real model ∂L/∂X carries on into the ResNet, the joint projection and, through z, the CVAE encoder.",
      },
      {
        update: true,
        on: ["enc-project", "dec-project", "E", "head"],
        title: "Update: every weight takes a small step against its gradient",
        equations: [
          {
            items: [
              mat("U<sub>V</sub>", p.UV),
              `− ${RATE} ×`,
              grad("∂L/∂U<sub>V</sub>", g.UV),
              "=",
              mat("U<sub>V</sub>", updated.UV, { caption: "new" }),
              { big: `L: ${c.loss.toFixed(3)} → ${lossAfter.toFixed(3)}` },
            ],
          },
        ],
        note: `<strong>Update.</strong> One of the nine learned matrices is shown; the others are treated the same way. With the new weights, the loss on this sample goes from ${c.loss.toFixed(3)} to ${lossAfter.toFixed(3)}. Press <em>Apply update</em> to keep them and send the sample through again.`,
      },
    ];
  }

  // --- drawing ----------------------------------------------------------------

  function recompute() {
    cache = forward(params, SAMPLE.X, SAMPLE.target);
    grads = backward(params, SAMPLE.X, cache);
    steps = buildSteps();
  }

  function render() {
    const current = steps[index];
    const forwardCount = steps.filter((s) => !s.back && !s.update).length;
    const backCount = steps.filter((s) => s.back).length;

    strip.setAttribute("class", current.back ? "bwd" : "fwd");
    strip.querySelectorAll("[data-block]").forEach((block) => block.classList.toggle("on", current.on.includes(block.dataset.block)));
    dirLabel.textContent = current.update ? "update the weights" : current.back ? "← backward pass: the same blocks, in reverse" : "forward pass →";

    titleEl.textContent = current.title;
    detail.textContent = "";
    current.equations.forEach((equation) => detail.appendChild(drawEquation(equation)));
    noteEl.innerHTML = current.note;
    scaleNote.textContent = current.back || current.update
      ? "Gradients: each matrix is coloured against its own largest entry."
      : "Values: colour saturates at ±1.5. Attention weights run from 0 to 1.";

    progress.textContent = current.update
      ? "Update"
      : current.back
        ? `Backward ${index - forwardCount + 1} of ${backCount}`
        : `Forward ${index + 1} of ${forwardCount}`;
    prevButton.disabled = index === 0;
    nextButton.textContent = current.update ? "Apply update" : "Next";
    historyEl.textContent =
      history.length > 1 ? `Loss on this sample after each update: ${history.map((loss) => loss.toFixed(3)).join(" → ")}` : "";
  }

  const player = Viz.ticker(2600, () => {
    if (index >= steps.length - 1) {
      stop();
      return;
    }
    index += 1;
    render();
  });

  function stop() {
    player.stop();
    playButton.textContent = "Play";
  }

  prevButton.addEventListener("click", () => {
    stop();
    index = Math.max(0, index - 1);
    render();
  });
  nextButton.addEventListener("click", () => {
    stop();
    if (steps[index].update) {
      params = step(params, grads, RATE);
      recompute();
      history.push(cache.loss);
      index = 0;
    } else {
      index += 1;
    }
    render();
  });
  playButton.addEventListener("click", () => {
    if (player.running) {
      stop();
      return;
    }
    if (index >= steps.length - 1) index = 0;
    render();
    player.start();
    playButton.textContent = "Pause";
  });

  recompute();
  history.push(cache.loss);
  render();
});
