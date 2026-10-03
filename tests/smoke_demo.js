// Headless smoke test for the AgentArena demo viewer.
// Loads data/demo.html in jsdom, executes its scripts, clicks through every
// tab, game switch, replay, step control and the reveal toggle, and fails on
// any uncaught JS error or missing critical content.
const fs = require("fs");
const path = require("path");
const { JSDOM } = require(path.join("/tmp/jscheck/node_modules", "jsdom"));

const file = process.argv[2] || "/home/user/agentarena/data/demo.html";
const html = fs.readFileSync(file, "utf8");

const errors = [];
const dom = new JSDOM(html, {
  runScripts: "dangerously",
  pretendToBeVisual: true,
  virtualConsole: new (require(path.join("/tmp/jscheck/node_modules", "jsdom")).VirtualConsole)()
    .on("jsdomError", e => errors.push("jsdomError: " + e.message))
    .on("error", (...a) => errors.push("console.error: " + a.join(" "))),
});
const { window } = dom;
window.addEventListener("error", e => errors.push("window.error: " + e.message));
const doc = window.document;

function clickAll(selector, label) {
  const els = [...doc.querySelectorAll(selector)];
  if (!els.length) errors.push(`no elements for ${label} (${selector})`);
  for (const el of els) {
    try { el.click(); } catch (e) { errors.push(`click failed on ${label}: ${e.message}`); }
  }
  return els.length;
}

// initial render checks
const chips = doc.querySelectorAll("#metachips .chip").length;
if (chips < 3) errors.push("meta chips missing");

// tab traversal
const navButtons = [...doc.querySelectorAll("nav button")].filter(b => !b.closest(".gameswitch"));
for (const b of navButtons) {
  b.click();
  const active = doc.querySelector("section.tab.active");
  if (!active || !active.innerHTML.length) errors.push(`tab ${b.textContent} rendered empty`);
}
// game switch traversal on every tab
const gameButtons = [...doc.querySelectorAll(".gameswitch button")];
if (gameButtons.length < 2) errors.push("game switch missing games");
for (const gb of gameButtons) {
  for (const tb of navButtons) {
    gb.click(); tb.click();
    const active = doc.querySelector("section.tab.active");
    if (!active || active.innerHTML.trim().length < 100)
      errors.push(`tab ${tb.textContent} x game ${gb.textContent} looks empty`);
  }
}

// replay interaction: for each game, open replays tab, click every replay item,
// step to the end, toggle reveal, use play/pause.
for (const gb of gameButtons) {
  gb.click();
  navButtons.find(b => b.textContent.includes("Replays")).click();
  const items = [...doc.querySelectorAll(".ritem")];
  if (!items.length) { errors.push(`no replays for ${gb.textContent}`); continue; }
  for (let i = 0; i < items.length; i++) {
    doc.querySelectorAll(".ritem")[i].click();
    const slider = doc.querySelector("#stepslider");
    if (!slider) { errors.push("no step slider"); break; }
    const max = parseInt(slider.max || "0", 10);
    for (let s = 0; s <= max; s++) {
      slider.value = String(s);
      slider.dispatchEvent(new window.Event("input", { bubbles: true }));
    }
    const reveal = doc.querySelector('.controls input[type=checkbox]');
    reveal.checked = true; reveal.dispatchEvent(new window.Event("change", { bubbles: true }));
    reveal.checked = false; reveal.dispatchEvent(new window.Event("change", { bubbles: true }));
    const play = doc.querySelector("#playbtn");
    play.click(); play.click();
    const body = doc.querySelector("#rbody");
    if (!body || body.innerHTML.trim().length < 50)
      errors.push(`replay ${i} body empty for ${gb.textContent}`);
  }
}
// leaderboard bar sanity
navButtons.find(b => b.textContent.includes("Leaderboard")).click();
if (!doc.querySelectorAll(".elobar .fill").length) errors.push("no elo bars rendered");
navButtons.find(b => b.textContent.includes("matrix")).click();
if (!doc.querySelectorAll(".matrix .cell").length) errors.push("no matrix cells rendered");

if (errors.length) {
  console.error("FAIL:\n" + errors.map(e => " - " + e).join("\n"));
  process.exit(1);
}
console.log("PASS: demo.html rendered all tabs/games/replays with no JS errors");
console.log(`nav buttons: ${navButtons.length}, games: ${gameButtons.length}, chips: ${chips}`);
