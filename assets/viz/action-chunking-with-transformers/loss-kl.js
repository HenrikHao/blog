// The regularisation term for a single latent dimension: KL between the
// encoder's Gaussian and the standard normal prior, split into its two costs.
Viz.mount("loss-kl", (root) => {
  const BETA = 10;
  const BAR_FULL = 5; // nats that fill the bar
  const X_RANGE = 5;
  const Y_MAX = 1.4;
  const MARGIN = { left: 12, right: 12, top: 8, bottom: 22 };
  const HEIGHT = 190;
  // Mean |mu| and mean sigma of the real checkpoint's posterior (out/findings.json).
  const OURS = { mu: 0.001, sigma: 1.0 };

  const chartEl = root.querySelector("[data-chart]");
  const readout = root.querySelector("[data-readout]");
  const muInput = root.querySelector("[data-mu]");
  const sigmaInput = root.querySelector("[data-sigma]");
  const muOut = root.querySelector("[data-mu-out]");
  const sigmaOut = root.querySelector("[data-sigma-out]");
  const [meanBar, spreadBar] = root.querySelectorAll("[data-bar] i");

  const state = { mu: Number(muInput.value), sigma: Number(sigmaInput.value) };
  let width = 0;

  const density = (x, mu, sigma) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

  function draw() {
    const { mu, sigma } = state;
    const meanCost = (mu * mu) / 2;
    const spreadCost = (sigma * sigma - Math.log(sigma * sigma) - 1) / 2;
    const kl = meanCost + spreadCost;

    muOut.textContent = mu.toFixed(2);
    sigmaOut.textContent = sigma.toFixed(2);
    meanBar.style.width = `${Math.min(1, meanCost / BAR_FULL) * 100}%`;
    spreadBar.style.width = `${Math.min(1, spreadCost / BAR_FULL) * 100}%`;
    const digits = kl < 0.01 ? 6 : 3;
    readout.innerHTML =
      `KL = <b>${kl.toFixed(digits)} nats</b> for this dimension: ${meanCost.toFixed(digits)} from the mean and ${spreadCost.toFixed(digits)} from the width. ` +
      `With β = ${BETA} it adds <b>${(BETA * kl).toFixed(kl < 0.01 ? 5 : 2)}</b> to the loss. ` +
      `The latent has 32 such dimensions and their costs are summed.`;

    chartEl.textContent = "";
    if (!width) return;
    const svg = Viz.svg(
      "svg",
      { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img", "aria-label": "Bell curves of the standard normal prior and of the encoder's posterior for one latent dimension" },
      chartEl
    );
    const floor = HEIGHT - MARGIN.bottom;
    const x = (value) => MARGIN.left + ((value + X_RANGE) / (2 * X_RANGE)) * (width - MARGIN.left - MARGIN.right);
    const y = (value) => floor - (Math.min(value, Y_MAX) / Y_MAX) * (floor - MARGIN.top);

    const curve = (m, s, name) => {
      const points = [];
      for (let i = 0; i <= 200; i++) {
        const value = -X_RANGE + (i / 200) * 2 * X_RANGE;
        points.push(`${x(value).toFixed(1)},${y(density(value, m, s)).toFixed(1)}`);
      }
      Viz.svg("path", { class: `wash ${name}`, d: `M${x(-X_RANGE)},${floor}L${points.join("L")}L${x(X_RANGE)},${floor}Z` }, svg);
      Viz.svg("path", { class: `curve ${name}`, d: `M${points.join("L")}` }, svg);
    };
    curve(0, 1, "prior");
    curve(mu, sigma, "post");

    const axis = Viz.svg("g", { class: "ax" }, svg);
    Viz.svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: floor, y2: floor }, axis);
    for (let tick = -4; tick <= 4; tick += 2) {
      const text = Viz.svg("text", { x: x(tick), y: floor + 15, "text-anchor": "middle" }, axis);
      text.textContent = tick;
    }
    const name = Viz.svg("text", { x: width - MARGIN.right, y: floor + 15, "text-anchor": "end" }, axis);
    name.textContent = "z";
  }

  muInput.addEventListener("input", () => {
    state.mu = Number(muInput.value);
    draw();
  });
  sigmaInput.addEventListener("input", () => {
    state.sigma = Number(sigmaInput.value);
    draw();
  });
  root.querySelector('[data-preset="ours"]').addEventListener("click", () => {
    Object.assign(state, OURS);
    muInput.value = 0;
    sigmaInput.value = 1;
    draw();
  });

  Viz.onResize(chartEl, (next) => {
    width = next;
    draw();
  });
  draw();
});
