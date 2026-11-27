// Shared top bar: Overview · Replay demo · Walkthrough · Debug. Include with <script src="/static/nav.js"></script>
// as the first thing in <body>.
(function () {
  const SECTIONS = [
    { label: "Overview", href: "/overview", match: p => p === "/overview" },
    { label: "Replay demo", href: "/replay", match: p => p === "/replay" },
    { label: "Walkthrough", href: "/walkthrough", match: p => p === "/walkthrough" },
    // The IVO view is switched off for now (see TODOS.md). To bring it back: restore this line and the /ivo route in app.py.
    // { label: "IVO", href: "/ivo", match: p => p === "/ivo" },
    { label: "Debug", href: "/debug", match: p => p.startsWith("/debug") || p === "/live" },
  ];
  const style = document.createElement("style");
  style.textContent = `
    /* centred reading column on every page; the bar's items line up with it */
    body > *:not(.topnav) { max-width: 1080px; margin-left: auto; margin-right: auto; }
    .topnav { display: flex; align-items: center; gap: 4px; margin: -24px -28px 20px;
              padding: 0 calc(28px + max(0px, (100% - 1080px) / 2));
              height: 44px; border-bottom: 1px solid var(--ring); background: var(--surface);
              position: sticky; top: 0; z-index: 10; font-size: 13.5px; }
    .topnav .brand { font-weight: 600; margin-right: 18px; color: var(--ink); }
    .topnav a { color: var(--ink2); text-decoration: none; padding: 5px 10px; border-radius: 6px; }
    .topnav a:hover { background: var(--grid); }
    .topnav a.on { color: var(--ink); background: var(--grid); font-weight: 600; }
  `;
  document.head.appendChild(style);
  const nav = document.createElement("nav");
  nav.className = "topnav";
  // Design versions live under a URL prefix (/v1, /v2, /old; none = the default), served by the same pages
  // (app.py strips it; static/theme/variant.js picks the look). Links between pages keep the prefix.
  const PREFIX = (location.pathname.match(/^\/(v1|v2|old)(?=\/|$)/) || [""])[0];
  const here = location.pathname.slice(PREFIX.length) || "/";
  nav.innerHTML = `<span class="brand">TWC verification node</span>` + SECTIONS.map(s =>
    `<a href="${PREFIX + s.href}" class="${s.match(here) ? "on" : ""}">${s.label}</a>`).join("");
  document.body.prepend(nav);
  if (PREFIX) {
    const PAGES = /^\/(overview|replay|walkthrough|debug|live)?(?=$|[#?])/;
    const fix = a => { const h = a && a.getAttribute("href"); if (h && h[0] === "/" && h[1] !== "/" && PAGES.test(h)) a.setAttribute("href", PREFIX + h); };
    // links in the page as written, and any the page draws later, when they are followed
    document.addEventListener("DOMContentLoaded", () => document.querySelectorAll("a[href^='/']").forEach(fix));
    document.addEventListener("click", e => fix(e.target.closest && e.target.closest("a[href^='/']")), true);
  }
})();
