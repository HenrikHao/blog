Viz.mount("architecture", (root) => {
  const notes = {
    train:
      "<strong>Training.</strong> The CVAE encoder sees the demonstrated chunk and squeezes it into z. The policy has to rebuild that chunk from the observation plus z.",
    infer:
      "<strong>Inference.</strong> The CVAE encoder is thrown away and z is pinned to 0, the mean of the prior. What is left is a deterministic map from observation to action chunk.",
  };
  const buttons = root.querySelectorAll("[data-mode]");
  const note = root.querySelector("[data-note]");

  function setMode(mode) {
    root.classList.toggle("is-infer", mode === "infer");
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
    note.innerHTML = notes[mode];
  }

  buttons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
});
