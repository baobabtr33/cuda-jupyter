/* cuda-jupyter front end: Jupyter-style cells, CUDA pane, terminal. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  // ------------------------------------------------------------------ state
  const nb = {
    cells: [],
    selected: 0,
    mode: "command",          // "command" | "edit"
    dirty: false,
    clipboard: null,
    metadata: {},
    cudaCell: null,           // cell whose CUDA is shown
    cudaTab: "code",
    lineNumbers: false,
    busy: 0,
  };
  let nextId = 1;
  let execChain = Promise.resolve();

  const container = $("#notebook-container");
  const notebookEl = $("#notebook");

  // ------------------------------------------------------------------ cells
  function normalizeCuda(c) {
    if (!c) return null;
    if (c.files) return c;
    if (c.code != null) return { files: [{ name: "kernel.cu", lang: "cpp", code: c.code, line_map: c.line_map || [] }], ops: c.ops || [], stats: c.stats || {} };
    return null;
  }

  function createCell(type = "code", source = "", opts = {}) {
    const cell = {
      id: nextId++,
      type,
      execCount: opts.execCount ?? null,
      outputs: opts.outputs || [],
      cuda: normalizeCuda(opts.cuda),
      running: false,
      rendered: type === "markdown" && opts.rendered !== false && source.trim() !== "",
    };
    const root = el("div", "cell " + (type === "code" ? "code_cell" : "text_cell"));
    const input = el("div", "input");
    const prompt = el("div", "prompt input_prompt");
    const inner = el("div", "inner_cell");
    const area = el("div", "input_area");
    const md = el("div", "text_cell_render");
    inner.append(area, md);
    input.append(prompt, inner);
    const outWrap = el("div", "output_wrapper");
    root.append(input, outWrap);

    const cm = CodeMirror(area, {
      value: source,
      mode: type === "code" ? { name: "python", version: 3, singleLineStringErrors: false } : "markdown",
      theme: "ipython",
      indentUnit: 4,
      lineNumbers: nb.lineNumbers,
      matchBrackets: true,
      autoCloseBrackets: true,
      viewportMargin: Infinity,
      lineWrapping: type === "markdown",
      extraKeys: {
        "Shift-Enter": () => runSelected("advance"),
        "Ctrl-Enter": () => runSelected("stay"),
        "Cmd-Enter": () => runSelected("stay"),
        "Alt-Enter": () => runSelected("insert"),
        Esc: () => commandMode(),
        Tab: (c) => (c.somethingSelected() ? c.execCommand("indentMore") : c.replaceSelection("    ")),
        "Shift-Tab": (c) => c.execCommand("indentLess"),
        "Ctrl-/": "toggleComment",
        "Cmd-/": "toggleComment",
        Up: (c) => (c.getCursor().line === 0 && !c.somethingSelected() ? moveSelection(-1, true) : CodeMirror.Pass),
        Down: (c) => (c.getCursor().line === c.lastLine() && !c.somethingSelected() ? moveSelection(1, true) : CodeMirror.Pass),
      },
    });
    cm.on("focus", () => { select(indexOf(cell), false); setMode("edit"); });
    cm.on("change", () => markDirty());
    cm.on("cursorActivity", () => {
      if (nb.cudaCell === cell && nb.mode === "edit") linkFromPython(cell, cm.getCursor().line + 1, true);
    });

    root.addEventListener("mousedown", (e) => {
      if (e.target.closest(".cuda-badge")) return;
      const i = indexOf(cell);
      if (i !== nb.selected) select(i, false);
      if (!e.target.closest(".CodeMirror")) setMode("command");
    });
    md.addEventListener("dblclick", () => { unrender(cell); editSelected(); });

    Object.assign(cell, { root, prompt, area, md, outWrap, cm, inner });
    renderPrompt(cell);
    renderOutputs(cell);
    if (cell.rendered) renderMarkdown(cell);
    else if (type === "markdown") root.classList.add("unrendered");
    renderBadge(cell);
    return cell;
  }

  function indexOf(cell) { return nb.cells.indexOf(cell); }
  function current() { return nb.cells[nb.selected]; }

  function insertCell(index, cell) {
    nb.cells.splice(index, 0, cell);
    const ref = container.children[index] || null;
    container.insertBefore(cell.root, ref);
    cell.cm.refresh();
    markDirty();
    return cell;
  }

  function removeCell(index) {
    const [cell] = nb.cells.splice(index, 1);
    cell.root.remove();
    if (nb.cudaCell === cell) showCuda(null);
    markDirty();
    return cell;
  }

  function renderPrompt(cell) {
    if (cell.type !== "code") { cell.prompt.innerHTML = "&nbsp;"; return; }
    const n = cell.running ? "*" : (cell.execCount ?? " ");
    cell.prompt.textContent = `In [${n}]:`;
  }

  function renderBadge(cell) {
    let badge = cell.inner.querySelector(".cuda-badge");
    if (cell.type !== "code" || (!cell.cuda && !cell.running)) { badge?.remove(); return; }
    if (!badge) {
      badge = el("button", "cuda-badge");
      badge.title = "Show the CUDA translation of this cell";
      badge.addEventListener("click", (e) => { e.stopPropagation(); showCuda(cell, false); });
      cell.inner.append(badge);
    }
    badge.classList.toggle("running", cell.running);
    badge.classList.toggle("active", nb.cudaCell === cell && !cell.running);
    if (cell.running) badge.textContent = "tracing...";
    else {
      const s = cell.cuda.stats || {};
      badge.textContent = `CUDA · ${s.kernels ?? 0} kernel${s.kernels === 1 ? "" : "s"}`;
    }
  }

  function renderMarkdown(cell) {
    const src = cell.cm.getValue();
    cell.md.innerHTML = src.trim() ? marked.parse(src) : '<p class="muted">Type Markdown and LaTeX: <i>α²</i></p>';
    cell.rendered = true;
    cell.root.classList.remove("unrendered");
  }

  function unrender(cell) {
    if (cell.type !== "markdown") return;
    cell.rendered = false;
    cell.root.classList.add("unrendered");
    cell.cm.refresh();
  }

  // ------------------------------------------------------------------ outputs
  function renderOutputs(cell) {
    cell.outWrap.innerHTML = "";
    for (const out of cell.outputs) {
      const area = el("div", "output_area");
      const prompt = el("div", "prompt");
      const sub = el("div", "output_subarea");
      if (out.output_type === "execute_result") {
        prompt.classList.add("output_prompt");
        prompt.textContent = `Out[${out.execution_count ?? " "}]:`;
        sub.textContent = joinText(out.data?.["text/plain"]);
      } else if (out.output_type === "stream") {
        sub.textContent = joinText(out.text);
        if (out.name === "stderr") sub.classList.add("output_stderr");
      } else if (out.output_type === "error") {
        sub.classList.add("output_error");
        renderTraceback(sub, out);
      } else if (out.data?.["text/plain"]) {
        sub.textContent = joinText(out.data["text/plain"]);
      }
      area.append(prompt, sub);
      cell.outWrap.append(area);
    }
  }

  function renderTraceback(sub, out) {
    const tb = (out.traceback || []).join("");
    const lines = tb.split("\n");
    for (const line of lines) {
      const div = el("div");
      const m = line.match(/^(\s*File ")(.+?)(", line \d+.*)$/);
      if (m) {
        div.append(m[1]);
        div.append(el("span", "tb-file", m[2]));
        div.append(m[3]);
      } else if (line.startsWith(out.ename)) {
        div.append(el("span", "ename", out.ename));
        div.append(line.slice(out.ename.length));
      } else {
        div.textContent = line || "​";
      }
      sub.append(div);
    }
  }

  const joinText = (t) => (Array.isArray(t) ? t.join("") : t || "");

  // ------------------------------------------------------------------ selection & modes
  function select(i, focusNotebook = true) {
    if (!nb.cells.length) return;
    i = Math.max(0, Math.min(nb.cells.length - 1, i));
    current()?.root.classList.remove("selected", "editing");
    nb.selected = i;
    const cell = current();
    cell.root.classList.add("selected");
    if (nb.mode === "edit") cell.root.classList.add("editing");
    $("#cell_type").value = cell.type;
    if (cell.cuda && nb.cudaCell !== cell && !cell.running) showCuda(cell, false);
    if (focusNotebook) scrollIntoViewIfNeeded(cell.root);
  }

  function setMode(mode) {
    nb.mode = mode;
    current()?.root.classList.toggle("editing", mode === "edit");
    if (mode === "command") clearPythonLink();
  }

  function commandMode() {
    setMode("command");
    current()?.cm.getInputField().blur();
    notebookEl.focus({ preventScroll: true });
  }

  function editSelected() {
    const cell = current();
    if (!cell) return;
    if (cell.type === "markdown" && cell.rendered) unrender(cell);
    setMode("edit");
    cell.cm.focus();
  }

  function moveSelection(delta, enterEdit = false) {
    const i = nb.selected + delta;
    if (i < 0 || i >= nb.cells.length) return;
    setMode("command");
    select(i);
    if (enterEdit) {
      const cell = current();
      editSelected();
      const line = delta > 0 ? 0 : cell.cm.lastLine();
      cell.cm.setCursor({ line, ch: delta > 0 ? 0 : cell.cm.getLine(line).length });
    }
  }

  function scrollIntoViewIfNeeded(node) {
    const r = node.getBoundingClientRect();
    const p = notebookEl.getBoundingClientRect();
    if (r.top < p.top + 10 || r.bottom > p.bottom - 10) {
      node.scrollIntoView({ block: r.height > p.height ? "start" : "nearest", behavior: "smooth" });
    }
  }

  // ------------------------------------------------------------------ execution
  function runSelected(after) {
    const cell = current();
    if (!cell) return;
    runCell(cell).catch(() => {});
    if (after === "advance") {
      if (nb.selected === nb.cells.length - 1) insertCell(nb.cells.length, createCell("code"));
      setMode("command");
      select(nb.selected + 1);
      if (cell.type === "code" && current().cm.getValue() === "") editSelected();
      else commandMode();
    } else if (after === "insert") {
      insertCell(nb.selected + 1, createCell("code"));
      select(nb.selected + 1);
      editSelected();
    } else {
      commandMode();
    }
  }

  function runCell(cell) {
    if (cell.type === "markdown") { renderMarkdown(cell); return Promise.resolve(); }
    cell.running = true;
    renderPrompt(cell);
    renderBadge(cell);
    if (nb.cudaCell === cell || !nb.cudaCell || current() === cell) showTracing(cell);
    const code = cell.cm.getValue();
    const job = execChain.then(() => execute(cell, code));
    execChain = job.catch(() => {});
    return job;
  }

  async function execute(cell, code) {
    setBusy(+1);
    cell.outputs = [];
    renderOutputs(cell);
    let r;
    try {
      r = await runJob(code);
    } catch (err) {
      r = { error: { ename: "ConnectionError", evalue: String(err),
        traceback: [`ConnectionError: lost contact with the server (${err.message || err}).\n`,
          "Is the server (run.sh / run_t4) still running? The cell may still have run there.\n"] } };
    } finally {
      setBusy(-1);
    }
    cell.running = false;
    cell.execCount = r.execution_count ?? null;
    const outs = [];
    if (r.stdout) outs.push({ output_type: "stream", name: "stdout", text: r.stdout });
    if (r.stderr) outs.push({ output_type: "stream", name: "stderr", text: r.stderr });
    if (r.error) outs.push({ output_type: "error", ...r.error });
    if (r.result != null) {
      outs.push({ output_type: "execute_result", execution_count: r.execution_count, data: { "text/plain": r.result }, metadata: {} });
    }
    cell.outputs = outs;
    cell.cuda = normalizeCuda(r.cuda);
    renderPrompt(cell);
    renderOutputs(cell);
    renderBadge(cell);
    if (cell.cuda) showCuda(cell, true);
    else if (nb.cudaCell === cell || nb.tracingCell === cell) showCuda(cell, false);
    markDirty();
    if (r.error) throw new Error(r.error.ename);
  }

  // Cells run as server-side jobs: start one, then long-poll for its result.
  // Network errors are retried, so a slow cell or a tunnel hiccup doesn't lose the result.
  async function runJob(code) {
    const id = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now() + Math.random());
    await withRetry(() => fetchJSON("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, code }),
    }));
    for (;;) {
      const st = await withRetry(() => fetchJSON(`/api/job/${id}?wait=20`));
      if (st.done) return st.result;
    }
  }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    if (res.status === 404) throw Object.assign(new Error("job not found on server (was it restarted?)"), { fatal: true });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function withRetry(fn) {
    const started = Date.now();
    for (let attempt = 0; ; attempt++) {
      try {
        const v = await fn();
        if (attempt) notify("Reconnected", 1500);
        return v;
      } catch (err) {
        if (err.fatal || Date.now() - started > 180000) throw err;
        notify("Connection lost, retrying...");
        await new Promise((r) => setTimeout(r, Math.min(1000 * 2 ** attempt, 8000)));
      }
    }
  }

  async function runRange(from, to) {
    for (let i = from; i < to; i++) {
      const cell = nb.cells[i];
      try { await runCell(cell); } catch { return; }
    }
  }

  function setBusy(delta) {
    nb.busy += delta;
    const ind = $("#kernel_indicator");
    ind.classList.toggle("busy", nb.busy > 0);
    ind.title = nb.busy > 0 ? "Kernel busy" : "Kernel idle";
    document.title = (nb.busy > 0 ? "(*) " : "") + "cuda_jupyter_tour - cuda-jupyter";
  }

  async function restartKernel() {
    notify("Restarting kernel...");
    await execChain;
    await fetch("/api/restart", { method: "POST" });
    for (const c of nb.cells) { if (c.type === "code") { c.execCount = null; renderPrompt(c); } }
    notify("Kernel restarted", 1500);
  }

  // ------------------------------------------------------------------ CUDA pane
  const cudaCode = $("#cuda-code");
  const cudaTrace = $("#cuda-trace");
  const cudaGpu = $("#cuda-gpu");
  const VIEW_TABS = ["trace", "gpu"];  // tabs that are not files
  const cudaEmpty = $("#cuda-empty");
  let revealJob = 0;

  function cellLabel(cell) {
    return cell.execCount != null ? `cell In [${cell.execCount}]` : `cell #${indexOf(cell) + 1}`;
  }

  function showTracing(cell) {
    nb.tracingCell = cell;
    nb.cudaCell = null;
    nb.cells.forEach(renderBadge);
    GpuView.stop();
    cudaCode.hidden = true;
    cudaTrace.hidden = true;
    cudaGpu.hidden = true;
    cudaEmpty.hidden = false;
    cudaEmpty.classList.add("tracing");
    cudaEmpty.innerHTML = '<div class="empty-mark">&lt;&lt;&lt; &gt;&gt;&gt;</div><p>Running PyTorch and tracing ATen ops...</p>';
    $("#cuda-cell-label").textContent = `cell #${indexOf(cell) + 1} · running`;
    $("#cuda-stats").innerHTML = "";
  }

  function showCuda(cell, animate) {
    nb.tracingCell = null;
    nb.cudaCell = cell && cell.cuda ? cell : null;
    nb.cells.forEach(renderBadge);
    cudaEmpty.classList.remove("tracing");
    if (!nb.cudaCell) {
      GpuView.stop();
      cudaCode.hidden = true;
      cudaTrace.hidden = true;
      cudaGpu.hidden = true;
      cudaEmpty.hidden = false;
      cudaEmpty.innerHTML = cell
        ? '<div class="empty-mark">&lt;&lt;&lt; &gt;&gt;&gt;</div><p>This cell dispatched no tensor ops,</p><p class="muted">so there is nothing to translate to CUDA.</p>'
        : '<div class="empty-mark">&lt;&lt;&lt; &gt;&gt;&gt;</div><p>Run a code cell (<kbd>Shift</kbd> + <kbd>Enter</kbd>).</p><p class="muted">PyTorch, JAX and TensorFlow ops are traced and translated into CUDA C++; Numba, Triton and CuPy kernels show the PTX / CUDA they compiled to.</p>';
      $("#cuda-cell-label").textContent = cell ? `${cellLabel(cell)} · no tensor ops` : "no cell run yet";
      $("#cuda-stats").innerHTML = "";
      renderFileTabs(null);
      return;
    }
    const cuda = cell.cuda;
    $("#cuda-cell-label").textContent = cellLabel(cell);
    renderStats(cuda.stats || {});
    if (!(nb.cudaTab === "trace" && (cuda.ops || []).length) && !(nb.cudaTab === "gpu" && (cuda.launches || []).length)) nb.cudaTab = 0;
    renderFileTabs(cuda);
    if (!VIEW_TABS.includes(nb.cudaTab)) renderCode(cuda.files[nb.cudaTab], animate);
    renderTrace(cuda);
    applyTab();
  }

  function renderFileTabs(cuda) {
    const box = $("#cuda-files");
    box.innerHTML = "";
    const files = cuda ? cuda.files : [{ name: "kernel.cu", lang: "cpp" }];
    files.forEach((f, i) => {
      const b = el("button", "pane-tab");
      b.dataset.tab = String(i);
      const icon = el("span", "file-icon" + (f.lang === "ptx" ? " ptx" : ""), f.lang === "ptx" ? "ptx" : "cu");
      b.append(icon, el("span", null, f.name));
      b.title = f.name;
      b.addEventListener("click", () => selectTab(i));
      box.append(b);
    });
    $("#trace-tab").hidden = !cuda || !(cuda.ops || []).length;
    $("#gpu-tab").hidden = !cuda || !(cuda.launches || []).length;
  }

  function selectTab(tab) {
    nb.cudaTab = tab;
    $("#cuda-body").scrollTop = 0;
    if (!VIEW_TABS.includes(tab) && nb.cudaCell) renderCode(nb.cudaCell.cuda.files[tab], false);
    applyTab();
  }

  function renderStats(s) {
    const items = [
      ["ops", "ops"], ["kernels", "kernels"], ["launches", "launches"],
      ["views", "views"], ["memcpy", "memcpy"], ["fallback", "untranslated"],
    ];
    const box = $("#cuda-stats");
    box.innerHTML = "";
    for (const [k, label] of items) {
      if (!s[k] && k !== "kernels") continue;
      const chip = el("span", "stat" + (k === "fallback" ? " warn" : ""));
      const b = el("b", null, String(s[k] ?? 0));
      chip.append(b, " " + label);
      box.append(chip);
    }
  }

  const escapeHtml = (t) => t.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  // Minimal PTX highlighter: comments, directives, registers, labels, numbers, opcodes.
  function highlightPtx(line) {
    if (!line.trim()) return "";
    const ci = line.indexOf("//");
    const code = ci >= 0 ? line.slice(0, ci) : line;
    const comment = ci >= 0 ? `<span class="hljs-comment">${escapeHtml(line.slice(ci))}</span>` : "";
    let html = escapeHtml(code)
      .replace(/(%[a-z]+[0-9]*|%[a-z]+\.[xyz])/g, '<span class="ptx-reg">$1</span>')
      .replace(/(^|\s)(\.[a-z_][\w.]*)/g, '$1<span class="hljs-meta">$2</span>')
      .replace(/^(\s*)([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)(?=\s)/, '$1<span class="hljs-keyword">$2</span>')
      .replace(/^(\s*)(\$?[A-Za-z_][\w$]*:)/, '$1<span class="hljs-title">$2</span>')
      .replace(/\b(0[xXfFdD][0-9a-fA-F]+|\d+)\b(?![^<]*>)/g, '<span class="hljs-number">$1</span>');
    return html + comment;
  }

  function highlightLine(line, lang) {
    if (lang === "ptx") return highlightPtx(line);
    if (!line.trim()) return "";
    try {
      let html = hljs.highlight(line, { language: "cpp", ignoreIllegals: true }).value;
      // make kernel launches stand out: name<<<grid, block>>>
      html = html.replace(/(\w+)(&lt;&lt;&lt;)/, '<span class="launch">$1</span>$2');
      return html;
    } catch {
      return escapeHtml(line);
    }
  }

  function renderCode(cuda, animate) {
    const job = ++revealJob;
    cudaCode.innerHTML = "";
    cudaEmpty.hidden = true;
    const lines = cuda.code.split("\n");
    const map = cuda.line_map || [];
    const build = (i) => {
      const row = el("div", "cl");
      const py = map[i] || 0;
      if (py) row.dataset.py = py;
      const g = el("span", "gutter", String(i + 1));
      const s = el("span", "src");
      s.innerHTML = highlightLine(lines[i], cuda.lang) || " ";
      row.append(g, s);
      return row;
    };
    if (!animate) {
      const frag = document.createDocumentFragment();
      for (let i = 0; i < lines.length; i++) frag.append(build(i));
      cudaCode.append(frag);
      return;
    }
    // Reveal: stream the translation in over ~0.7s so it reads as "generated after the run".
    const perFrame = Math.max(6, Math.ceil(lines.length / 40));
    let i = 0;
    $("#cuda-body").scrollTop = 0;
    const step = () => {
      if (job !== revealJob) return;
      const frag = document.createDocumentFragment();
      for (let k = 0; k < perFrame && i < lines.length; k++, i++) {
        const row = build(i);
        row.classList.add("reveal");
        frag.append(row);
      }
      cudaCode.append(frag);
      if (i < lines.length) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderTrace(cuda) {
    cudaTrace.innerHTML = "";
    const t = el("table", "trace-table");
    t.innerHTML = "<thead><tr><th>#</th><th>line</th><th>op</th><th>inputs</th><th>output</th><th>x</th></tr></thead>";
    const tb = el("tbody");
    (cuda.ops || []).forEach((op, i) => {
      const tr = el("tr", op.kind === "view" ? "view" : "");
      tr.dataset.py = op.line;
      const td = (cls, text) => { const c = el("td", cls, text); tr.append(c); return c; };
      td("muted", String(i + 1));
      td("", op.line ? `L${op.line}` : "");
      const opCell = td("op", op.op);
      if (op.phase === "backward") opCell.append(" ", el("span", "bwd", "bwd"));
      td("shape", op.ins.join(", "));
      td("shape", op.outs.join(", "));
      td("count", op.count > 1 ? `x${op.count}` : "");
      tb.append(tr);
    });
    t.append(tb);
    cudaTrace.append(t);
  }

  function applyTab() {
    document.querySelectorAll("#cuda-pane .pane-tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === String(nb.cudaTab)));
    if (!nb.cudaCell) return;
    cudaCode.hidden = VIEW_TABS.includes(nb.cudaTab);
    cudaTrace.hidden = nb.cudaTab !== "trace";
    cudaGpu.hidden = nb.cudaTab !== "gpu";
    if (nb.cudaTab === "gpu") {
      GpuView.render(cudaGpu, nb.cudaCell.cuda, nb.gpu, { nvcc: nb.nvcc, onGoto: gotoLaunch });
    } else {
      GpuView.stop();
    }
  }

  function currentFile() {
    const c = nb.cudaCell?.cuda;
    if (!c) return null;
    return c.files[VIEW_TABS.includes(nb.cudaTab) ? 0 : nb.cudaTab] || c.files[0];
  }

  // GPU tab -> the launch line in the generated file
  function gotoLaunch(name, line) {
    const files = nb.cudaCell?.cuda.files || [];
    const i = files.findIndex((f) => f.name === name);
    if (i < 0) return;
    selectTab(i);
    const row = cudaCode.children[line - 1];
    if (!row) return;
    row.scrollIntoView({ block: "center" });
    row.classList.add("flash");
    setTimeout(() => row.classList.remove("flash"), 1600);
  }

  // cross-highlighting between CUDA lines and Python lines
  let linkedRows = [];
  let linkedCm = null;

  function clearLinks() {
    linkedRows.forEach((r) => r.classList.remove("linked"));
    linkedRows = [];
    clearPythonLink();
  }

  function clearPythonLink() {
    if (linkedCm) {
      const { cm, line } = linkedCm;
      cm.removeLineClass(line, "background", "cm-linked-line");
      linkedCm = null;
    }
  }

  function linkCudaRows(py) {
    linkedRows.forEach((r) => r.classList.remove("linked"));
    linkedRows = py ? [...document.querySelectorAll(`#cuda-pane [data-py="${py}"]`)] : [];
    linkedRows.forEach((r) => r.classList.add("linked"));
    return linkedRows;
  }

  function linkPython(cell, py) {
    clearPythonLink();
    if (!py || py > cell.cm.lineCount()) return;
    cell.cm.addLineClass(py - 1, "background", "cm-linked-line");
    linkedCm = { cm: cell.cm, line: py - 1 };
  }

  function linkFromPython(cell, py, scroll) {
    const rows = linkCudaRows(py);
    const first = rows.find((r) => r.offsetParent !== null);
    if (!scroll || !first) return;
    const rb = first.getBoundingClientRect(), bb = $("#cuda-body").getBoundingClientRect();
    if (rb.top < bb.top || rb.bottom > bb.bottom) first.scrollIntoView({ block: "center" });
  }

  $("#cuda-body").addEventListener("mouseover", (e) => {
    const row = e.target.closest("[data-py]");
    if (!row || !nb.cudaCell) return;
    const py = +row.dataset.py;
    linkCudaRows(py);
    linkPython(nb.cudaCell, py);
  });
  $("#cuda-body").addEventListener("mouseleave", () => {
    clearLinks();
    const cell = nb.cudaCell;
    if (cell && nb.mode === "edit" && current() === cell) linkFromPython(cell, cell.cm.getCursor().line + 1, false);
  });
  $("#cuda-body").addEventListener("click", (e) => {
    const row = e.target.closest("[data-py]");
    if (!row || !nb.cudaCell) return;
    const cell = nb.cudaCell;
    const py = +row.dataset.py;
    select(indexOf(cell));
    scrollIntoViewIfNeeded(cell.root);
    if (py <= cell.cm.lineCount()) {
      setMode("edit");
      cell.cm.focus();
      cell.cm.setCursor({ line: py - 1, ch: 0 });
    }
  });

  $("#trace-tab").addEventListener("click", () => selectTab("trace"));
  $("#gpu-tab").addEventListener("click", () => selectTab("gpu"));

  $("#cuda-copy").addEventListener("click", async () => {
    if (!nb.cudaCell) return;
    try { await navigator.clipboard.writeText(currentFile().code); notify("Copied " + currentFile().name, 1500); }
    catch { notify("Clipboard unavailable", 1500); }
  });
  $("#cuda-download").addEventListener("click", () => downloadCuda());

  function downloadCuda() {
    const cell = nb.cudaCell || current();
    if (!cell?.cuda) { notify("Run the cell first", 1500); return; }
    const f = nb.cudaCell === cell ? currentFile() : cell.cuda.files[0];
    download(f.name, f.code, "text/plain");
  }

  function download(name, text, type) {
    const a = el("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  // ------------------------------------------------------------------ notebook I/O
  function toIpynb() {
    return {
      cells: nb.cells.map((c) => {
        const source = c.cm.getValue();
        if (c.type === "markdown") return { cell_type: "markdown", metadata: {}, source };
        return {
          cell_type: "code",
          execution_count: c.execCount,
          metadata: c.cuda ? { cuda: c.cuda } : {},
          outputs: c.outputs,
          source,
        };
      }),
      metadata: nb.metadata,
      nbformat: 4,
      nbformat_minor: 5,
    };
  }

  function loadIpynb(data) {
    nb.cells.forEach((c) => c.root.remove());
    nb.cells = [];
    nb.metadata = data.metadata || {};
    for (const c of data.cells || []) {
      const src = joinText(c.source);
      const type = c.cell_type === "markdown" ? "markdown" : "code";
      const cell = createCell(type, src, {
        execCount: c.execution_count,
        outputs: c.outputs,
        cuda: c.metadata?.cuda,
      });
      nb.cells.push(cell);
      container.append(cell.root);
    }
    if (!nb.cells.length) { const c = createCell("code"); nb.cells.push(c); container.append(c.root); }
    nb.cells.forEach((c) => c.cm.refresh());
    nb.selected = 0;
    select(0, false);
    nb.dirty = false;
    showCuda(null);
  }

  async function save() {
    notify("Saving notebook...");
    try {
      await fetch("/api/notebook", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toIpynb()),
      });
      nb.dirty = false;
      $("#checkpoint_status").textContent = "Last Checkpoint: " + new Date().toLocaleTimeString();
      $("#autosave_status").textContent = "";
      notify("Notebook saved", 1500);
    } catch {
      notify("Save failed", 3000);
    }
  }

  function markDirty() {
    nb.dirty = true;
    $("#autosave_status").textContent = "(unsaved changes)";
  }

  setInterval(() => { if (nb.dirty && !nb.busy) save(); }, 120000);

  let notifyTimer = 0;
  function notify(msg, ms) {
    const n = $("#notification_area");
    n.textContent = msg;
    n.classList.add("show");
    clearTimeout(notifyTimer);
    if (ms) notifyTimer = setTimeout(() => { n.textContent = ""; n.classList.remove("show"); }, ms);
  }

  // ------------------------------------------------------------------ commands
  function changeType(type) {
    const cell = current();
    if (!cell || cell.type === type) return;
    const fresh = createCell(type, cell.cm.getValue(), { rendered: false });
    const i = nb.selected;
    removeCell(i);
    insertCell(i, fresh);
    select(i);
    if (type === "markdown") unrender(fresh);
  }

  const commands = {
    save,
    "download-ipynb": () => download("cuda_jupyter_tour.ipynb", JSON.stringify(toIpynb(), null, 1), "application/json"),
    "download-cu": downloadCuda,
    revert: async () => {
      const ex = await fetch("/api/example").then((r) => r.json()).catch(() => null);
      if (ex) { loadIpynb(ex); markDirty(); }
    },
    cut: () => { if (!nb.cells.length) return; nb.clipboard = snapshot(current()); removeCell(nb.selected); ensureCell(); select(nb.selected); },
    copy: () => { nb.clipboard = snapshot(current()); notify("Cell copied", 1000); },
    paste: () => {
      if (!nb.clipboard) return;
      const c = createCell(nb.clipboard.type, nb.clipboard.source, { rendered: false });
      insertCell(nb.selected + 1, c);
      select(nb.selected + 1);
    },
    delete: () => { if (!nb.cells.length) return; removeCell(nb.selected); ensureCell(); select(nb.selected); },
    "move-up": () => moveCell(-1),
    "move-down": () => moveCell(1),
    "toggle-cuda": () => { document.body.classList.toggle("no-cuda"); store.set("hideCuda", document.body.classList.contains("no-cuda")); fitTerminal(); },
    "toggle-terminal": () => {
      document.body.classList.toggle("no-terminal");
      store.set("hideTerminal", document.body.classList.contains("no-terminal"));
      if (!document.body.classList.contains("no-terminal")) { fitTerminal(); term.focus(); }
    },
    "toggle-lines": () => {
      nb.lineNumbers = !nb.lineNumbers;
      nb.cells.forEach((c) => c.cm.setOption("lineNumbers", nb.lineNumbers));
    },
    "insert-above": () => { insertCell(nb.selected, createCell("code")); select(nb.selected); },
    "insert-below": () => { insertCell(nb.selected + 1, createCell("code")); select(nb.selected + 1); },
    run: () => runSelected("stay"),
    "run-advance": () => runSelected("advance"),
    "run-insert": () => runSelected("insert"),
    "run-all": () => runRange(0, nb.cells.length),
    "run-above": () => runRange(0, nb.selected),
    "run-below": () => runRange(nb.selected, nb.cells.length),
    "to-code": () => changeType("code"),
    "to-markdown": () => changeType("markdown"),
    "clear-output": () => clearOutput(current()),
    "clear-all": () => nb.cells.forEach(clearOutput),
    restart: restartKernel,
    "restart-clear": async () => { await restartKernel(); nb.cells.forEach(clearOutput); },
    "restart-run": async () => { await restartKernel(); nb.cells.forEach(clearOutput); runRange(0, nb.cells.length); },
    shortcuts: showShortcuts,
    about: () => openModal("About", `<p>A Jupyter-style notebook that runs PyTorch and shows, next to every cell, the CUDA C++ its ATen ops translate to.</p><p class="muted" id="about-env"></p>`),
  };

  function snapshot(cell) { return { type: cell.type, source: cell.cm.getValue() }; }
  function ensureCell() { if (!nb.cells.length) insertCell(0, createCell("code")); }

  function moveCell(delta) {
    const i = nb.selected, j = i + delta;
    if (j < 0 || j >= nb.cells.length) return;
    const cell = nb.cells[i];
    nb.cells.splice(i, 1);
    nb.cells.splice(j, 0, cell);
    cell.root.remove();
    container.insertBefore(cell.root, container.children[j] || null);
    nb.selected = j;
    cell.cm.refresh();
    scrollIntoViewIfNeeded(cell.root);
    markDirty();
  }

  function clearOutput(cell) {
    if (!cell || cell.type !== "code") return;
    cell.outputs = [];
    cell.cuda = null;
    renderOutputs(cell);
    renderBadge(cell);
    if (nb.cudaCell === cell) showCuda(null);
    markDirty();
  }

  function run(cmd) { const f = commands[cmd]; if (f) f(); }

  document.querySelectorAll("[data-cmd]").forEach((n) => n.addEventListener("click", (e) => {
    if (n.disabled) return;
    e.stopPropagation();
    closeMenus();
    run(n.dataset.cmd);
    if (n.closest("#toolbar") && !["run-advance", "insert-below"].includes(n.dataset.cmd)) commandMode();
  }));

  $("#cell_type").addEventListener("change", (e) => { changeType(e.target.value); commandMode(); });
  $("#notebook-end").addEventListener("click", () => {
    const last = nb.cells[nb.cells.length - 1];
    if (!last || last.cm.getValue() !== "") insertCell(nb.cells.length, createCell("code"));
    select(nb.cells.length - 1);
    editSelected();
  });

  // menus
  function closeMenus() { document.querySelectorAll(".menu.open").forEach((m) => m.classList.remove("open")); }
  document.querySelectorAll(".menu").forEach((m) => {
    m.addEventListener("click", (e) => {
      if (e.target.closest(".dropdown")) return;
      const open = m.classList.contains("open");
      closeMenus();
      if (!open) m.classList.add("open");
    });
    m.addEventListener("mouseenter", () => {
      if (document.querySelector(".menu.open") && !m.classList.contains("open")) { closeMenus(); m.classList.add("open"); }
    });
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });

  // modal
  function openModal(title, html) {
    $("#modal-title").textContent = title;
    $("#modal-body").innerHTML = html;
    $("#modal").hidden = false;
    fetch("/api/info").then((r) => r.json()).then((i) => {
      const n = $("#about-env");
      if (n) n.textContent = `torch ${i.torch} · Python ${i.python} · device: ${i.device}`;
    }).catch(() => {});
  }
  const closeModal = () => { $("#modal").hidden = true; };
  $("#modal-close").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });

  function showShortcuts() {
    const rows = (pairs) => pairs.map(([k, d]) => `<kbd>${k}</kbd><span>${d}</span>`).join("");
    openModal("Keyboard shortcuts", `<div class="shortcuts">
      <h4>Command mode (press Esc)</h4>
      ${rows([["Enter", "edit cell"], ["Shift-Enter", "run, select below"], ["Ctrl-Enter", "run"], ["Alt-Enter", "run, insert below"],
        ["A / B", "insert cell above / below"], ["D, D", "delete cell"], ["X / C / V", "cut / copy / paste cell"],
        ["Y / M", "to code / markdown"], ["↑ / K, ↓ / J", "select previous / next"], ["L", "toggle line numbers"],
        ["0, 0", "restart kernel"], ["Ctrl-S", "save"], ["Ctrl-`", "toggle terminal"]])}
      <h4>Edit mode (press Enter)</h4>
      ${rows([["Esc", "command mode"], ["Shift-Enter", "run, select below"], ["Tab / Shift-Tab", "indent / dedent"], ["Ctrl-/", "toggle comment"]])}
      <h4>CUDA pane</h4>
      ${rows([["hover", "highlight the Python line that produced a CUDA line"], ["click", "jump to that Python line"]])}
    </div>`);
  }

  // ------------------------------------------------------------------ keyboard (command mode)
  let lastKey = "", lastKeyAt = 0;
  document.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
    if (e.ctrlKey && e.key === "`") { e.preventDefault(); run("toggle-terminal"); return; }
    if (!$("#modal").hidden) { if (e.key === "Escape") closeModal(); return; }
    if (nb.mode !== "command") return;
    const t = e.target;
    if (t.closest && (t.closest("#terminal") || t.closest("input, select, textarea"))) return;

    const k = e.key;
    const twice = (key) => { const ok = lastKey === key && Date.now() - lastKeyAt < 600; lastKey = ok ? "" : key; lastKeyAt = Date.now(); return ok; };
    if (k === "Enter" && e.shiftKey) { e.preventDefault(); runSelected("advance"); return; }
    if (k === "Enter" && mod) { e.preventDefault(); runSelected("stay"); return; }
    if (k === "Enter" && e.altKey) { e.preventDefault(); runSelected("insert"); return; }
    if (mod || e.altKey) return;
    switch (k) {
      case "Enter": e.preventDefault(); editSelected(); break;
      case "ArrowUp": case "k": e.preventDefault(); select(nb.selected - 1); break;
      case "ArrowDown": case "j": e.preventDefault(); select(nb.selected + 1); break;
      case "a": run("insert-above"); break;
      case "b": run("insert-below"); break;
      case "d": if (twice("d")) run("delete"); break;
      case "0": if (twice("0")) run("restart"); break;
      case "x": run("cut"); break;
      case "c": run("copy"); break;
      case "v": run("paste"); break;
      case "m": run("to-markdown"); break;
      case "y": run("to-code"); break;
      case "l": run("toggle-lines"); break;
      case "h": run("shortcuts"); break;
      default: return;
    }
  });

  // ------------------------------------------------------------------ splitters
  function dragSplit(handle, onMove) {
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      const move = (ev) => onMove(ev);
      const up = () => {
        handle.classList.remove("dragging");
        document.body.classList.remove("resizing");
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        fitTerminal();
        nb.cells.forEach((c) => c.cm.refresh());
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  }
  dragSplit($("#vsplit"), (e) => {
    const ws = $("#workspace").getBoundingClientRect();
    const w = Math.max(280, Math.min(ws.width - 360, ws.right - e.clientX));
    document.documentElement.style.setProperty("--cuda-w", w + "px");
    store.set("cudaW", w);
  });
  dragSplit($("#hsplit"), (e) => {
    const m = $("#main").getBoundingClientRect();
    const h = Math.max(80, Math.min(m.height - 160, m.bottom - e.clientY));
    document.documentElement.style.setProperty("--term-h", h + "px");
    store.set("termH", h);
    fitTerminal();
  });
  const cw = store.get("cudaW", null), th = store.get("termH", null);
  if (cw) document.documentElement.style.setProperty("--cuda-w", cw + "px");
  if (th) document.documentElement.style.setProperty("--term-h", th + "px");
  if (store.get("hideCuda", false)) document.body.classList.add("no-cuda");
  if (store.get("hideTerminal", false)) document.body.classList.add("no-terminal");

  // ------------------------------------------------------------------ terminal
  const term = new Terminal({
    fontFamily: 'Menlo, Monaco, "Courier New", monospace',
    fontSize: 12.5,
    lineHeight: 1.2,
    cursorBlink: true,
    scrollback: 5000,
    allowProposedApi: false,
    theme: {
      background: "#1e1e1e", foreground: "#d4d4d4", cursor: "#aeafad", selectionBackground: "#264f78",
      black: "#000000", red: "#cd3131", green: "#0dbc79", yellow: "#e5e510", blue: "#2472c8",
      magenta: "#bc3fbc", cyan: "#11a8cd", white: "#e5e5e5", brightBlack: "#666666", brightRed: "#f14c4c",
      brightGreen: "#23d18b", brightYellow: "#f5f543", brightBlue: "#3b8eea", brightMagenta: "#d670d6",
      brightCyan: "#29b8db", brightWhite: "#e5e5e5",
    },
  });
  const fitAddon = new FitAddon.FitAddon();
  term.loadAddon(fitAddon);
  term.open($("#terminal"));
  let ws = null;
  let wsAlive = false;

  function fitTerminal() {
    if (document.body.classList.contains("no-terminal")) return;
    try { fitAddon.fit(); } catch { /* not visible yet */ }
  }

  function sendResize() {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
  }

  function connectTerminal() {
    if (ws) { try { ws.close(); } catch { /* ignore */ } }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/terminal`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsAlive = true; fitTerminal(); sendResize(); };
    ws.onmessage = (e) => term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data));
    ws.onclose = () => {
      if (!wsAlive) return;
      wsAlive = false;
      term.write("\r\n\x1b[90m[shell exited, press Enter to start a new one]\x1b[0m\r\n");
    };
  }

  term.onData((d) => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "input", data: d }));
    else if (d === "\r") { term.reset(); connectTerminal(); }
  });
  term.onResize(sendResize);
  term.textarea?.addEventListener("focus", () => setMode("command"));
  $("#term-clear").addEventListener("click", () => { term.clear(); term.focus(); });
  $("#term-restart").addEventListener("click", () => { wsAlive = false; term.reset(); connectTerminal(); term.focus(); });
  $("#term-hide").addEventListener("click", () => run("toggle-terminal"));
  window.addEventListener("resize", () => { fitTerminal(); });
  new ResizeObserver(() => fitTerminal()).observe($("#terminal"));

  // ------------------------------------------------------------------ boot
  async function boot() {
    try {
      const info = await fetch("/api/info").then((r) => r.json());
      $("#env_info").textContent = `torch ${info.torch} · ${info.device}`;
      $("#term-title").textContent = `${info.shell} · ${info.device}`;
      nb.gpu = info.gpu;
      nb.nvcc = info.nvcc;
    } catch { /* server info optional */ }
    const data = await fetch("/api/notebook").then((r) => r.json());
    loadIpynb(data);
    connectTerminal();
    fitTerminal();
    commandMode();
  }
  boot();
})();
