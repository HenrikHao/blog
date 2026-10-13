// Small shared helpers for the interactive figures. Loaded (deferred) on pages
// with `viz: true`; each figure script calls Viz.mount("<name>", init).
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const Viz = {
    get reducedMotion() {
      return motion.matches;
    },

    mount(name, init) {
      document.querySelectorAll(`[data-viz="${name}"]`).forEach((root) => {
        // A figure used twice on a page loads its script twice; start each one once.
        if (root.classList.contains("is-live")) return;
        try {
          init(root);
          root.classList.add("is-live");
        } catch (err) {
          console.error(`viz "${name}" failed to start`, err);
        }
      });
    },

    svg(tag, attrs, parent) {
      const el = document.createElementNS(SVG_NS, tag);
      for (const key in attrs) el.setAttribute(key, attrs[key]);
      if (parent) parent.appendChild(el);
      return el;
    },

    // URLs of the figure's data files (see the viz shortcode), keyed by extension.
    assets(root) {
      return JSON.parse(root.dataset.assets || "{}");
    },

    // Runs callback once, shortly before el scrolls into view. For lazy loading.
    whenNear(el, callback) {
      const observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((entry) => entry.isIntersecting)) {
            observer.disconnect();
            callback();
          }
        },
        { rootMargin: "800px 0px" }
      );
      observer.observe(el);
    },

    // "#rrggbb" custom property as [r, g, b].
    rgb(el, name) {
      const hex = Viz.token(el, name).replace("#", "");
      return [0, 2, 4].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
    },

    // Resolved value of a CSS custom property, for canvas drawing.
    token(el, name) {
      return getComputedStyle(el).getPropertyValue(name).trim();
    },

    onTheme(callback) {
      new MutationObserver(callback).observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });
    },

    onResize(el, callback) {
      let width = 0;
      new ResizeObserver((entries) => {
        const next = Math.round(entries[0].contentRect.width);
        if (next !== width) {
          width = next;
          callback(next);
        }
      }).observe(el);
    },

    // Runs callback once, the first time a good part of el is on screen.
    whenVisible(el, callback) {
      const observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((entry) => entry.isIntersecting)) {
            observer.disconnect();
            callback();
          }
        },
        { threshold: 0.35 }
      );
      observer.observe(el);
    },

    // Deterministic PRNG (mulberry32) so a figure looks the same on every load.
    rng(seed) {
      let a = seed >>> 0;
      return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },

    // Calls onTick every `interval` ms until stopped. Pauses itself when the
    // page is hidden.
    ticker(interval, onTick) {
      let timer = null;
      const api = {
        get running() {
          return timer !== null;
        },
        start() {
          if (timer === null) timer = setInterval(onTick, interval);
        },
        stop() {
          clearInterval(timer);
          timer = null;
        },
      };
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) api.stop();
      });
      return api;
    },

    clamp(value, lo, hi) {
      return Math.min(hi, Math.max(lo, value));
    },
  };

  window.Viz = Viz;
})();
