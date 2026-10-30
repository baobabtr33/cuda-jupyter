/* GPU tab: shows how one kernel launch maps onto the device - grid -> SMs (waves),
   then inside one SM: warp schedulers, cores, register file and shared memory. */
(() => {
  "use strict";

  const BLOCK_COLORS = ["#76b900", "#4fc1ff", "#c586c0", "#e5c07b", "#f07178", "#56b6c2", "#d19a66", "#9cdcfe"];
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const kb = (b) => (b >= 1024 * 1024 ? `${(b / 1048576).toFixed(b % 1048576 ? 1 : 0)} MB`
    : b >= 1024 ? `${(b / 1024).toFixed(b % 1024 ? 1 : 0)} KB` : `${b} B`);
  const prod = (d) => d[0] * d[1] * d[2];
  const dimText = (d) => (d[2] > 1 ? `${d[0]}×${d[1]}×${d[2]}` : d[1] > 1 ? `${d[0]}×${d[1]}` : `${d[0]}`);

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // Rough register counts for the generated templates when nvcc is not available.
  function estimateRegs(name) {
    if (/gemm|conv/.test(name)) return 40;
    if (/rand|dropout/.test(name)) return 32;  // curand Philox state
    if (/softmax|norm|reduce|argmax|nll|mse|pool/.test(name)) return 28;
    return 16;
  }

  function regsFor(launch, cuda) {
    if (launch.regs) return { regs: launch.regs, src: launch.regs_src };
    const r = st(cuda).res?.kernels?.[launch.kernel];
    if (r?.regs != null) return { regs: Math.max(r.regs, 1), src: "ptxas", spill: r.spill, smem: r.smem };
    return { regs: estimateRegs(launch.kernel), src: "estimate" };
  }

  // The occupancy calculator, simplified from NVIDIA's: how many blocks of this launch
  // fit on one SM at the same time, and which resource runs out first.
  function occupancy(gpu, launch, regs, smem) {
    const threads = prod(launch.block);
    const warpsPerBlock = Math.ceil(threads / 32);
    const regsPerWarp = Math.ceil((regs * 32) / 256) * 256;
    const perPartition = gpu.regs_per_sm / gpu.partitions;
    const warpsByRegs = Math.floor(perPartition / regsPerWarp) * gpu.partitions;
    const smemPerBlock = smem ? Math.ceil((smem + (gpu.major >= 8 ? 1024 : 0)) / 256) * 256 : 0;
    const limits = [
      { key: "threads", label: "Warp slots", value: Math.floor(gpu.max_warps / warpsPerBlock),
        why: `${gpu.max_warps} warps per SM ÷ ${warpsPerBlock} warps per block` },
      { key: "regs", label: "Registers", value: Math.floor(warpsByRegs / warpsPerBlock),
        why: `${fmt(gpu.regs_per_sm)} regs ÷ (${regs} regs × ${warpsPerBlock * 32} threads)` },
      { key: "smem", label: "Shared memory", value: smemPerBlock ? Math.floor((gpu.smem_kb * 1024) / smemPerBlock) : Infinity,
        why: smemPerBlock ? `${gpu.smem_kb} KB per SM ÷ ${kb(smemPerBlock)} per block` : "kernel uses no shared memory" },
      { key: "blocks", label: "Block slots", value: gpu.max_blocks, why: `hardware limit of ${gpu.max_blocks} blocks per SM` },
    ];
    const blocksPerSM = Math.max(0, Math.min(...limits.map((l) => l.value)));
    const limiter = limits.find((l) => l.value === blocksPerSM);
    return {
      threads, warpsPerBlock, regsPerWarp, smemPerBlock, limits, blocksPerSM, limiter,
      activeWarps: blocksPerSM * warpsPerBlock,
      occupancy: (blocksPerSM * warpsPerBlock) / gpu.max_warps,
    };
  }

  // ------------------------------------------------------------------ state
  // per-cell UI state (selected launch, wave, ptxas results), kept out of the saved notebook
  const state = new WeakMap();
  const st = (cuda) => { if (!state.has(cuda)) state.set(cuda, {}); return state.get(cuda); };
  let timer = null;
  let tick = 0;
  let view = null;  // {root, cuda, gpu, opts}

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function render(root, cuda, gpu, opts = {}) {
    stop();
    view = { root, cuda, gpu, opts };
    root.innerHTML = "";
    const launches = cuda.launches || [];
    if (!gpu) { root.append(el("p", "gpu-note", "Device information unavailable.")); return; }
    if (!launches.length) {
      root.append(deviceCard(gpu), el("p", "gpu-note", "This cell launched no kernels."));
      return;
    }
    if (st(cuda).sel == null || st(cuda).sel >= launches.length) {
      // start on the biggest launch: it is the most interesting one to look at
      st(cuda).sel = launches.reduce((best, l, i) => (prod(l.grid) * prod(l.block) > prod(launches[best].grid) * prod(launches[best].block) ? i : best), 0);
      st(cuda).wave = 0;
    }
    maybeFetchRegs(cuda, gpu, opts);
    root.append(deviceCard(gpu));
    if (st(cuda).res?.pending) root.append(working("compiling with nvcc --resource-usage for exact register counts", st(cuda).res.t0));
    root.append(launchList(cuda, gpu));
    const launch = launches[st(cuda).sel];
    const r = regsFor(launch, cuda);
    const smem = Math.max(launch.smem || 0, r.smem || 0);
    const occ = occupancy(gpu, launch, r.regs, smem);
    if (!occ.blocksPerSM) {
      root.append(el("p", "gpu-note warn", `This launch cannot run: a block needs more than one SM has (${occ.limiter.label.toLowerCase()}).`));
      return;
    }
    root.append(chipView(cuda, gpu, launch, occ), smView(cuda, gpu, launch, occ, r, smem), occupancyView(gpu, launch, occ, r));
  }

  function rerender() {
    if (view && view.root.isConnected) render(view.root, view.cuda, view.gpu, view.opts);
  }

  async function maybeFetchRegs(cuda, gpu, opts) {
    if (st(cuda).res || !opts.nvcc) return;
    const file = cuda.files.find((f) => f.lang === "cpp" && (cuda.launches || []).some((l) => l.file === f.name && !l.regs));
    if (!file) return;
    st(cuda).res = { pending: true, t0: performance.now() };
    try {
      const res = await fetch("/api/resources", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: file.code, arch: gpu.sm }),
      }).then((r) => r.json());
      st(cuda).res = res;
    } catch (e) {
      st(cuda).res = { error: String(e) };
    }
    if (view?.cuda === cuda) rerender();
  }

  // Animated "still working" line with a live elapsed time.
  function working(text, t0) {
    const box = el("div", "gpu-working");
    const mark = el("span", "empty-mark working");
    [..."<<<>>>"].forEach((c, i) => { const sp = el("span", null, c); sp.style.setProperty("--i", i); mark.append(sp); });
    const secs = el("span", "muted", "");
    box.append(mark, el("span", null, text), secs);
    const tickElapsed = () => {
      if (!box.isConnected && secs.textContent) return;
      secs.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
      setTimeout(tickElapsed, 100);
    };
    tickElapsed();
    return box;
  }

  // ------------------------------------------------------------------ sections
  function deviceCard(gpu) {
    const box = el("div", "gpu-device");
    const title = el("div", "gpu-device-name");
    title.append(el("b", null, gpu.name), el("span", "muted", ` ${gpu.family} · ${gpu.sm}`));
    const facts = el("div", "gpu-facts");
    const fact = (v, l) => { const f = el("span", "gpu-fact"); f.append(el("b", null, v), " " + l); facts.append(f); };
    fact(gpu.sms, "SMs");
    fact(fmt(gpu.sms * gpu.fp32), "FP32 cores");
    if (gpu.tc) fact(gpu.sms * gpu.tc, "tensor cores");
    fact(`${fmt(gpu.regs_per_sm / 1024)}K`, "registers / SM");
    fact(`${gpu.smem_kb} KB`, "shared mem / SM");
    if (gpu.l2) fact(kb(gpu.l2), "L2");
    fact(`${(gpu.total_memory / 2 ** 30).toFixed(0)} GB`, "DRAM");
    box.append(title, facts);
    if (gpu.assumed) box.append(el("div", "gpu-note", "No CUDA device on this machine: showing a Tesla T4 (the Colab GPU) for illustration."));
    return box;
  }

  function launchList(cuda, gpu) {
    const sec = section("Kernel launches in this cell", "Pick one to see how it runs on the GPU.");
    const list = el("div", "gpu-launches");
    (cuda.launches || []).forEach((l, i) => {
      const row = el("div", "gpu-launch" + (i === st(cuda).sel ? " active" : ""));
      if (l.py_line) row.dataset.py = l.py_line;
      row.append(
        el("span", "gl-line", l.py_line ? `L${l.py_line}` : ""),
        el("span", "gl-name", l.kernel),
        el("span", "gl-dims", `<<<${dimText(l.grid)}, ${dimText(l.block)}>>>`),
        el("span", "gl-count", l.repeat > 1 ? `×${l.repeat}` : ""),
      );
      row.title = `${l.op}  ·  ${fmt(prod(l.grid))} blocks × ${fmt(prod(l.block))} threads`;
      row.addEventListener("click", (e) => {
        e.stopPropagation();
        st(cuda).sel = i;
        st(cuda).wave = 0;
        rerender();
      });
      list.append(row);
    });
    sec.append(list);
    return sec;
  }

  function section(title, sub) {
    const s = el("section", "gpu-section");
    const h = el("h4", null, title);
    s.append(h);
    if (sub) s.append(el("p", "gpu-sub", sub));
    return s;
  }

  function chipView(cuda, gpu, launch, occ) {
    const blocks = prod(launch.grid);
    const perWave = occ.blocksPerSM * gpu.sms;
    const waves = Math.ceil(blocks / perWave);
    st(cuda).wave = Math.min(st(cuda).wave || 0, waves - 1);
    const sec = section("1 · The grid is split into blocks, blocks go to SMs");
    const busySMs = Math.min(blocks, gpu.sms);
    let story = `${fmt(blocks)} block${blocks > 1 ? "s" : ""} of ${fmt(occ.threads)} threads. ` +
      `One SM holds up to ${occ.blocksPerSM} of these blocks at a time, so ${gpu.sms} SMs run ${fmt(perWave)} blocks per wave` +
      (waves > 1 ? `: ${fmt(waves)} waves.` : ".");
    sec.append(el("p", "gpu-sub", story));
    if (busySMs < gpu.sms) sec.append(el("p", "gpu-callout", `Only ${busySMs} of ${gpu.sms} SMs get any work: this launch is too small to fill the GPU.`));
    else if (waves > 1 && blocks % perWave) {
      const last = blocks % perWave;
      sec.append(el("p", "gpu-callout", `The last wave has only ${fmt(last)} blocks, so part of the GPU idles at the end (the "tail effect").`));
    }

    const ctrl = el("div", "gpu-ctrl");
    const prev = el("button", "pane-btn", "◀");
    const next = el("button", "pane-btn", "▶");
    const play = el("button", "pane-btn", timer ? "Pause" : "Play");
    const label = el("span", "gpu-wave", "");
    ctrl.append(prev, label, next, play);
    sec.append(ctrl);

    const chip = el("div", "gpu-chip");
    chip.style.gridTemplateColumns = `repeat(${Math.ceil(Math.sqrt(gpu.sms * 2))}, 1fr)`;
    const tiles = [];
    for (let s = 0; s < gpu.sms; s++) {
      const t = el("div", "sm-tile" + (s === 0 ? " focus" : ""));
      t.append(el("span", "sm-label", `SM ${s}`));
      const slots = el("div", "sm-slots");
      for (let j = 0; j < occ.blocksPerSM; j++) slots.append(el("i"));
      t.append(slots);
      chip.append(t);
      tiles.push(t);
    }
    sec.append(chip);
    const mem = el("div", "gpu-mem");
    mem.append(el("span", null, `L2 cache ${gpu.l2 ? kb(gpu.l2) : ""}`), el("span", null, `DRAM ${(gpu.total_memory / 2 ** 30).toFixed(0)} GB  (tensors live here)`));
    sec.append(mem);
    sec.append(el("p", "gpu-foot", "Simplified: blocks are dealt round-robin; real hardware hands a new block to an SM as soon as one finishes."));

    const paint = () => {
      const w = st(cuda).wave;
      const n = Math.min(perWave, blocks - w * perWave);
      label.textContent = `wave ${w + 1} / ${fmt(waves)}  ·  blocks ${fmt(w * perWave)}–${fmt(w * perWave + n - 1)}`;
      tiles.forEach((t, s) => {
        const count = Math.floor(n / gpu.sms) + (s < n % gpu.sms ? 1 : 0);
        t.classList.toggle("idle", count === 0);
        [...t.querySelector(".sm-slots").children].forEach((c, j) => {
          c.className = j < count ? "on" : "";
          c.style.background = j < count ? BLOCK_COLORS[j % BLOCK_COLORS.length] : "";
        });
      });
      prev.disabled = w === 0;
      next.disabled = w >= waves - 1;
    };
    paint();
    const go = (w) => { st(cuda).wave = (w + waves) % waves; rerenderSM(); };
    prev.addEventListener("click", () => go(st(cuda).wave - 1));
    next.addEventListener("click", () => go(st(cuda).wave + 1));
    play.addEventListener("click", () => {
      if (timer) { stop(); play.textContent = "Play"; return; }
      play.textContent = "Pause";
      timer = setInterval(() => {
        if (!view?.root.isConnected || view.root.offsetParent === null) { stop(); return; }
        tick++;
        animateSchedulers();
        if (tick % 8 === 0 && waves > 1) go(st(cuda).wave + 1);
      }, 160);
    });
    // wave changes only redraw the chip and the SM, not the whole tab
    function rerenderSM() {
      paint();
      const old = view.root.querySelector(".gpu-sm-section");
      if (old) old.replaceWith(smView(cuda, gpu, launch, occ, regsFor(launch, cuda), Math.max(launch.smem || 0, regsFor(launch, cuda).smem || 0)));
    }
    return sec;
  }

  function blockCoord(launch, b) {
    const [gx, gy] = launch.grid;
    const x = b % gx, y = Math.floor(b / gx) % gy, z = Math.floor(b / (gx * gy));
    return launch.grid[2] > 1 ? `(${x},${y},${z})` : launch.grid[1] > 1 ? `(${x},${y})` : `${x}`;
  }

  function smView(cuda, gpu, launch, occ, r, smem) {
    const blocks = prod(launch.grid);
    const perWave = occ.blocksPerSM * gpu.sms;
    const w = st(cuda).wave || 0;
    const n = Math.min(perWave, blocks - w * perWave);
    const resident = [];  // block ids on SM 0 this wave
    for (let k = 0; k < n; k += gpu.sms) resident.push(w * perWave + k);
    const sec = section("2 · Inside one SM (SM 0)");
    sec.classList.add("gpu-sm-section");
    const wpb = occ.warpsPerBlock;
    const lanesLast = occ.threads - (wpb - 1) * 32;
    sec.append(el("p", "gpu-sub",
      `Each block is cut into warps of 32 threads (${fmt(occ.threads)} threads = ${wpb} warp${wpb > 1 ? "s" : ""}). ` +
      `The SM has ${gpu.partitions} warp schedulers; warps are spread over them. Every cycle each scheduler picks one ready warp ` +
      `and issues its next instruction to its ${gpu.fp32 / gpu.partitions} FP32 cores` +
      (gpu.fp32 / gpu.partitions < 32 ? `, so a 32-thread warp takes ${32 / (gpu.fp32 / gpu.partitions)} cycles per instruction.` : ".") +
      " While one warp waits on memory, the scheduler switches to another for free: that is why many resident warps help."));
    if (lanesLast < 32) sec.append(el("p", "gpu-callout", `The block size is not a multiple of 32: the last warp of each block runs with ${lanesLast} of 32 lanes active.`));

    const sm = el("div", "gpu-sm");
    sm.style.setProperty("--parts", gpu.partitions);
    const slotsPerPart = gpu.max_warps / gpu.partitions;
    const parts = [];
    for (let p = 0; p < gpu.partitions; p++) parts.push([]);
    resident.forEach((b, j) => {
      for (let wi = 0; wi < wpb; wi++) {
        const sw = j * wpb + wi;  // warp index within the SM
        parts[sw % gpu.partitions].push({ block: b, j, warp: wi, lanes: wi === wpb - 1 ? lanesLast : 32 });
      }
    });
    const regsPerPart = gpu.regs_per_sm / gpu.partitions;
    parts.forEach((warps, p) => {
      const part = el("div", "sm-part");
      part.append(el("div", "sm-part-title", `Warp scheduler ${p}`));
      const ws = el("div", "warp-slots");
      for (let k = 0; k < slotsPerPart; k++) {
        const wv = warps[k];
        const slot = el("div", "warp-slot" + (wv ? " on" : ""));
        if (wv) {
          slot.style.setProperty("--c", BLOCK_COLORS[wv.j % BLOCK_COLORS.length]);
          slot.append(el("span", null, `b${blockCoord(launch, wv.block)} w${wv.warp}`));
          const lanes = el("i", "lanes");
          lanes.style.width = `${(wv.lanes / 32) * 100}%`;
          slot.append(lanes);
          slot.title = `block ${blockCoord(launch, wv.block)}, warp ${wv.warp}: threads ${wv.warp * 32}–${wv.warp * 32 + wv.lanes - 1}`;
        }
        ws.append(slot);
      }
      part.append(ws);
      const units = el("div", "sm-units");
      const group = (n, cls, label) => {
        if (!n) return;
        const g = el("div", "unit-group");
        g.append(el("span", "unit-label", `${n} ${label}`));
        const cells = el("div", "unit-cells " + cls);
        for (let i = 0; i < n; i++) cells.append(el("i"));
        g.append(cells);
        units.append(g);
      };
      group(gpu.fp32 / gpu.partitions, "fp32", "FP32");
      group(gpu.int32 / gpu.partitions, "int32", "INT32");
      group(gpu.tc / gpu.partitions, "tc", gpu.tc / gpu.partitions > 1 ? "tensor cores" : "tensor core");
      part.append(units);
      const used = warps.length * occ.regsPerWarp;
      part.append(bar(`Registers`, used / regsPerPart, `${fmt(used)} / ${fmt(regsPerPart)}`));
      sm.append(part);
    });
    sec.append(sm);
    const smemUsed = resident.length * occ.smemPerBlock;
    sec.append(bar(`Shared memory / L1 · ${gpu.smem_kb} KB`, smemUsed / (gpu.smem_kb * 1024),
      smem ? `${kb(smemUsed)} used: ${resident.length} block${resident.length === 1 ? "" : "s"} × ${kb(occ.smemPerBlock)}` : "unused by this kernel (data comes straight from L2 / DRAM)"));

    const th = el("div", "gpu-thread");
    th.append(el("div", "gpu-thread-title", `One thread owns ${r.regs} registers` +
      (r.src === "estimate" ? (st(cuda).res?.pending ? " (estimated, nvcc running...)" : " (estimated)") : r.src ? ` (${r.src})` : "")));
    const cells = el("div", "reg-cells");
    for (let i = 0; i < Math.min(r.regs, 64); i++) cells.append(el("i", null, ""));
    if (r.regs > 64) cells.append(el("span", "muted", ` +${r.regs - 64}`));
    th.append(cells);
    th.append(el("p", "gpu-foot", `Its local variables (indices, loop counters, accumulators) live here: the fastest storage on the GPU. ` +
      `${occ.threads} threads × ${r.regs} = ${fmt(occ.threads * r.regs)} registers per block.` +
      (r.spill ? ` ${r.spill} bytes spill to local memory (slow).` : "")));
    sec.append(th);
    return sec;
  }

  function bar(label, frac, detail) {
    const b = el("div", "gpu-bar");
    const head = el("div", "gpu-bar-head");
    head.append(el("span", null, label), el("span", "muted", detail));
    const track = el("div", "gpu-bar-track");
    const fill = el("i");
    fill.style.width = `${Math.min(100, frac * 100).toFixed(1)}%`;
    track.append(fill);
    b.append(head, track);
    return b;
  }

  function occupancyView(gpu, launch, occ, r) {
    const sec = section("3 · Occupancy: what limits blocks per SM");
    const t = el("table", "gpu-occ");
    occ.limits.forEach((l) => {
      const tr = el("tr", l === occ.limiter ? "limit" : "");
      tr.append(el("td", null, l.label), el("td", "num", l.value === Infinity ? "–" : String(l.value)), el("td", "muted", l.why));
      t.append(tr);
    });
    sec.append(t);
    const pct = Math.round(occ.occupancy * 100);
    sec.append(el("p", "gpu-sub",
      `${occ.blocksPerSM} blocks × ${occ.warpsPerBlock} warps = ${occ.activeWarps} of ${gpu.max_warps} warp slots: ${pct}% occupancy, limited by ${occ.limiter.label.toLowerCase()}.` +
      (r.src === "estimate" ? " Register count is estimated; run on a machine with nvcc for real numbers." : "")));
    if (view?.opts.onGoto && launch.file) {
      const a = el("button", "pane-btn", `Show launch in ${launch.file}`);
      a.addEventListener("click", (e) => { e.stopPropagation(); view.opts.onGoto(launch.file, launch.line); });
      sec.append(a);
    }
    return sec;
  }

  // Playing: each scheduler issues from a different resident warp every tick.
  function animateSchedulers() {
    if (!view) return;
    view.root.querySelectorAll(".sm-part").forEach((part, p) => {
      const on = [...part.querySelectorAll(".warp-slot.on")];
      on.forEach((s) => s.classList.remove("issue"));
      const cores = part.querySelector(".unit-cells.fp32");
      if (!on.length) { cores?.classList.remove("busy"); return; }
      on[(tick + p) % on.length].classList.add("issue");
      cores?.classList.toggle("busy", tick % 2 === 0);
    });
  }

  window.GpuView = { render, stop };
})();
