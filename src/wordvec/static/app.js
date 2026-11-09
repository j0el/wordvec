"use strict";

const lang = location.pathname.split("/")[1] === "zh" ? "zh" : "en";
const $ = (id) => document.getElementById(id);
const consoleEl = $("console");
const input = $("line");

let state = null;       // {walk, step, total} of the active walkthrough, or null
const history = [];     // commands typed this session, for the up arrow
let historyPos = 0;
let busy = false;

// Chinese characters occupy two terminal cells; see .w in style.css.
const WIDE = /[ᄀ-ᅟ⺀-〾ぁ-㏿㐀-䶿一-鿿豈-﫿＀-｠￠-￦]/;

// Build DOM from text with createTextNode only, never innerHTML, so nothing the
// server (or a crafted input echoed back) sends can become markup.
function appendText(parent, text) {
  let run = "";
  const flush = () => { if (run) { parent.appendChild(document.createTextNode(run)); run = ""; } };
  for (const ch of text) {
    if (WIDE.test(ch)) {
      flush();
      const span = document.createElement("span");
      span.className = "w";
      span.textContent = ch;
      parent.appendChild(span);
    } else {
      run += ch;
    }
  }
  flush();
}

function block(text, className) {
  const pre = document.createElement("pre");
  if (className) pre.className = className;
  appendText(pre, text);
  consoleEl.appendChild(pre);
  return pre;
}

function echo(prompt, line) {
  const pre = document.createElement("pre");
  pre.className = "echo";
  appendText(pre, prompt + " ");
  const b = document.createElement("b");
  appendText(b, line);
  pre.appendChild(b);
  consoleEl.appendChild(pre);
}

function updateStepper() {
  const stepper = $("stepper");
  stepper.hidden = state === null;
  if (state === null) return;
  $("stepinfo").textContent = `${state.walk} — step ${state.step} of ${state.total}`;
  $("back").disabled = state.step <= 1;
  $("next").disabled = state.step >= state.total;
}

async function run(line, { show = true } = {}) {
  if (busy) return;
  busy = true;
  document.body.style.cursor = "progress";
  try {
    const response = await fetch(`/api/${lang}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ line, state }),
    });
    const data = await response.json();
    if (!response.ok) {
      block(data.error || "Request failed.", "error");
      return;
    }
    if (data.clear) consoleEl.replaceChildren();
    // Walkthrough steps replace the screen (like the terminal); other commands append.
    if (show && line.trim() && !data.clear) echo(info.prompt, line);
    const out = data.output.replace(/^\n+/, "").replace(/\n+$/, "\n");
    if (out.trim()) {
      const pre = block(out);
      // Show the top of new output, not the bottom: panels are longer than the window.
      consoleEl.scrollTop = data.clear ? 0 : Math.max(0, pre.offsetTop - consoleEl.offsetTop - 40);
    }
    // Pictures the step asked for (animations, diagrams) go under its text.
    for (const spec of data.figures || []) consoleEl.appendChild(Figures.render(spec));
    state = data.state;
    updateStepper();
  } catch (error) {
    block("Could not reach the server. It may be starting up — try again in a moment.", "error");
  } finally {
    busy = false;
    document.body.style.cursor = "";
    input.focus({ preventScroll: true });
  }
}

let info = { prompt: ">" };

async function init() {
  for (const a of document.querySelectorAll(".langs a")) {
    if (a.dataset.lang === lang) a.setAttribute("aria-current", "page");
  }
  document.documentElement.lang = lang === "zh" ? "zh" : "en";
  const response = await fetch(`/api/${lang}/info`);
  info = await response.json();
  document.title = info.title;
  $("title").textContent = info.title;
  $("promptlabel").textContent = info.prompt;

  for (const { label, command } of info.buttons) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => run(command));
    $("buttons").appendChild(button);
  }
  for (const example of info.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example;
    button.addEventListener("click", () => { input.value = ""; run(example); });
    $("examples").appendChild(button);
  }
  block("Pick a walkthrough above, click an example below, or type an equation.\nPress Enter on an empty line to advance a walkthrough.\n", "hint");
  appendText($("helptext"), info.banner.replace(/^\n+/, "").replace(/\n+$/, ""));
  input.focus({ preventScroll: true });
}

$("form").addEventListener("submit", (event) => {
  event.preventDefault();
  const line = input.value;
  if (line.trim()) { history.push(line); historyPos = history.length; }
  input.value = "";
  run(line);
});
$("next").addEventListener("click", () => run("next"));
$("back").addEventListener("click", () => run("back"));

input.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp" && history.length) {
    historyPos = Math.max(0, historyPos - 1);
    input.value = history[historyPos];
    event.preventDefault();
  } else if (event.key === "ArrowDown" && history.length) {
    historyPos = Math.min(history.length, historyPos + 1);
    input.value = history[historyPos] || "";
    event.preventDefault();
  }
});

init();
