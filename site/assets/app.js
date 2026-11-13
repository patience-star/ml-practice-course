/* 机器学习实战课 · 共享组件
   不依赖任何第三方库，双击 html 也能离线运行。
   页面里的数字和代码都来自 site/data/*.js —— 由 build.py 用真实 scikit-learn 跑出来，不是手写的。 */
const ML = (() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const get = p => p.split('.').reduce((o, k) => (o == null ? o : o[k]), window);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const FMT = {
    pct: v => (v * 100).toFixed(1) + '%',
    pct0: v => Math.round(v * 100) + '%',
    spct: v => (v > 0 ? '+' : '') + (v * 100).toFixed(1) + '%',
    f1: v => (+v).toFixed(1), f2: v => (+v).toFixed(2), f3: v => (+v).toFixed(3),
    int: v => Math.round(v).toLocaleString('zh-CN'),
    wan: v => (+v).toFixed(1) + ' 万',
    yuan: v => (v < 0 ? '−¥' : '¥') + Math.abs(Math.round(v)).toLocaleString('zh-CN'),
    raw: v => String(v),
  };
  const fmt = (v, f = 'raw') => (v == null || Number.isNaN(v) ? '—' : (FMT[f] || FMT.raw)(v));

  /* ---------- 本地进度（读写失败也不影响阅读） ---------- */
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem('mlcourse:' + k)) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem('mlcourse:' + k, JSON.stringify(v)); } catch { /* 隐私模式等：忽略 */ } },
  };
  const page = document.body.dataset.page || 'home';
  const prog = store.get(page, { quiz: {} });
  const leaks = store.get('leaks', {});
  const save = () => { store.set(page, prog); store.set('leaks', leaks); refresh(); };

  /* ---------- 数值绑定：<b data-v="CLF.auc_leak" data-f="f3"></b> ---------- */
  function bind(root = document) {
    $$('[data-v]', root).forEach(el => { if (!el.closest('.guess') || el.closest('.reveal')) el.textContent = fmt(get(el.dataset.v), el.dataset.f); });
  }

  /* ---------- 代码块：<div data-code="CLF.code.clean"></div> ---------- */
  const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b(def|return|for|in|if|else|elif|import|from|as|with|lambda|not|and|or|is|None|True|False|while|class)\b|\b(\d[\d_]*(?:\.\d+)?(?:e-?\d+)?)\b/g;
  function hl(src) {
    let out = '', last = 0;
    src.replace(TOKEN, (m, com, str, kw, num, off) => {
      out += esc(src.slice(last, off));
      const cls = com ? (com.includes('❌') ? 'bad' : com.includes('✅') ? 'good' : 'c') : str ? 's' : kw ? 'k' : 'n';
      out += `<span class="t-${cls}">${esc(m)}</span>`;
      last = off + m.length;
      return m;
    });
    return out + esc(src.slice(last));
  }
  const FILES = { REG: 'regression.py', CLF: 'classification.py', CLU: 'clustering.py' };
  function code(root = document) {
    $$('[data-code]', root).forEach(el => {
      const src = get(el.dataset.code);
      const [ns, , name] = el.dataset.code.split('.');
      el.classList.add('code');
      if (src == null) { el.textContent = '缺少代码片段：请先运行 python build.py'; return; }
      const n = src.split('\n').length, fold = n > 18;
      el.innerHTML = `<div class="code-bar"><span>ml/${FILES[ns]} · step:${name}</span><span class="row" style="gap:6px">${fold ? `<button class="more">展开全部 ${n} 行</button>` : ''}<button class="cp">复制</button></span></div><pre><code>${hl(src)}</code></pre>`;
      if (fold) {
        el.classList.add('folded');
        $('.more', el).onclick = e => { const f = el.classList.toggle('folded'); e.target.textContent = f ? `展开全部 ${n} 行` : '收起'; };
      }
      $('.cp', el).onclick = e => {
        const done = t => { e.target.textContent = t; setTimeout(() => (e.target.textContent = '复制'), 1500); };
        try {
          navigator.clipboard.writeText(src).then(() => done('已复制'), () => { selectCode(el); done('已选中，按 Ctrl+C'); });
        } catch { selectCode(el); done('已选中，按 Ctrl+C'); }
      };
    });
  }
  function selectCode(el) { const r = document.createRange(); r.selectNodeContents($('pre', el)); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }

  /* ---------- 切换面板：[data-tabs] 内的 [data-show] 按钮控制 [data-pane] ---------- */
  function tabs(root = document) {
    $$('[data-tabs]', root).forEach(box => {
      const own = sel => $$(sel, box).filter(x => x.closest('[data-tabs]') === box);
      const btns = own('[data-show]'), panes = own('[data-pane]');
      const show = name => {
        btns.forEach(b => b.classList.toggle('on', b.dataset.show === name));
        panes.forEach(p => (p.hidden = p.dataset.pane !== name));
        box.dispatchEvent(new CustomEvent('pane', { detail: name }));
      };
      btns.forEach(b => (b.onclick = () => {
        show(b.dataset.show);
        const lk = box.closest('.leak');
        if (lk && lk.dataset.id && !leaks[lk.dataset.id]) { leaks[lk.dataset.id] = 1; save(); }
      }));
      show((btns.find(b => b.classList.contains('on')) || btns[0])?.dataset.show);
    });
  }

  /* ---------- 小测验 ---------- */
  function quizzes(root = document) {
    $$('.quiz', root).forEach((q, i) => {
      const step = q.closest('.step');
      const id = q.dataset.id || `${step ? step.id : 'x'}-q${$$('.quiz', step || document).indexOf(q)}`;
      q.dataset.id = id;
      const btns = $$('.opts button', q), ans = +q.dataset.answer, ex = $('.explain', q);
      if (ex) ex.hidden = true;
      const reveal = pick => {
        btns.forEach((b, j) => { b.disabled = true; if (j === ans) b.classList.add('right'); else if (j === pick) b.classList.add('wrong'); });
        if (ex) ex.hidden = false;
      };
      if (id in prog.quiz) reveal(prog.quiz[id] ? ans : -1);
      btns.forEach((b, j) => (b.onclick = () => { prog.quiz[id] = j === ans ? 1 : 0; reveal(j); save(); }));
    });
  }

  /* ---------- 先猜后看 ---------- */
  function guesses(root = document) {
    $$('.guess', root).forEach(g => {
      const input = $('input', g), out = $('output', g), btn = $('.go', g), rev = $('.reveal', g);
      const gf = g.dataset.gf || g.dataset.f;
      const show = () => (out.textContent = fmt(+input.value, gf));
      input.oninput = show; show(); rev.hidden = true;
      btn.onclick = () => {
        const truth = get(g.dataset.v), guess = +input.value;
        const span = +input.max - +input.min, off = Math.abs(guess - truth) / span;
        $('.truth', rev).textContent = fmt(truth, g.dataset.f);
        $('.diff', rev).textContent = off < 0.04 ? '你猜得很准，直觉不错。' : `你猜的是 ${fmt(guess, gf)}，${guess > truth ? '高' : '低'}估了。`;
        rev.hidden = false; btn.disabled = true; input.disabled = true;
        bind(rev);
      };
    });
  }

  /* ---------- 排序小游戏 ---------- */
  function orders(root = document) {
    $$('.order', root).forEach(box => {
      const answer = box.dataset.answer.split(','), pool = $('.pool', box), slots = $('.slots', box), msg = $('.msg', box);
      const items = $$('button', pool);
      const check = () => {
        const got = $$('li', slots).map(li => li.dataset.id);
        msg.className = 'msg';
        if (got.length < answer.length) { msg.textContent = `已放入 ${got.length} / ${answer.length} 步`; return; }
        const wrong = got.map((g, i) => g !== answer[i]);
        $$('li', slots).forEach((li, i) => $('button', li).classList.toggle('wrong', wrong[i]));
        msg.innerHTML = wrong.some(Boolean) ? '<span class="verdict bad">顺序不对。</span> 标红的几步放错了位置，点它退回重新放。' : `<span class="verdict good">完全正确。</span> ${box.dataset.ok || ''}`;
      };
      items.forEach(b => (b.onclick = () => {
        const li = document.createElement('li'); li.dataset.id = b.dataset.id;
        li.innerHTML = `<button>${b.innerHTML}</button>`; b.hidden = true;
        $('button', li).onclick = () => { li.remove(); b.hidden = false; check(); };
        slots.append(li); check();
      }));
      check();
    });
  }

  /* ---------- 字段归类（哪些字段是泄漏？） ---------- */
  function classify(root = document) {
    $$('.classify', root).forEach(box => {
      const chips = $$('.chip', box), msg = $('.msg', box);
      chips.forEach(c => (c.onclick = () => { if (!box.classList.contains('checked')) c.classList.toggle('on'); }));
      $('.go', box).onclick = () => {
        box.classList.add('checked');
        let right = 0;
        chips.forEach(c => {
          const ok = c.classList.contains('on') === (c.dataset.leak === '1');
          right += ok;
          c.classList.remove('on'); c.classList.add(c.dataset.leak === '1' ? 'no' : 'ok');
          c.insertAdjacentHTML('beforeend', ` <small>· ${c.dataset.why}${ok ? '' : '（你判断错了）'}</small>`);
        });
        msg.innerHTML = `判断对了 <b>${right} / ${chips.length}</b> 个。红色 = 预测时点拿不到（泄漏），绿色 = 可以用。`;
        $('.go', box).disabled = true;
        const lk = box.closest('.leak');
        if (lk && lk.dataset.id) { leaks[lk.dataset.id] = 1; save(); }
      };
    });
  }

  /* ---------- 给老板的汇报 ---------- */
  function reports(root = document) {
    $$('.report', root).forEach(box => {
      $('.go', box).onclick = () => {
        let score = 0, total = 0;
        $$('label', box).forEach(l => {
          const on = $('input', l).checked, good = l.dataset.good === '1';
          l.className = on ? (good ? 'good' : 'bad') : good ? 'miss' : '';
          $('.fb', l).hidden = false;
          if (good) total++;
          if (on) score += good ? 1 : -1;
        });
        $('.msg', box).innerHTML = `得分 <b>${Math.max(score, 0)} / ${total}</b>。红框：不该对老板这样说；虚线绿框：漏掉的关键结论。`;
      };
      $$('.fb', box).forEach(f => (f.hidden = true));
    });
  }

  /* ---------- 找茬表格 ---------- */
  const cell = v => (v == null ? '<span class="na">NaN</span>' : typeof v === 'string' && /^\s|\s$/.test(v) ? esc(v.replace(/ /g, '␣')) : esc(v));
  function spot(el, { sample, flags, flag_cols: flagCols = [] }) {
    const F = new Map(flags.map(f => [`${f[0]}|${f[1]}`, f])), C = new Map(flagCols.map(f => [f[0], f]));
    const total = F.size + C.size, found = new Set();
    el.innerHTML = `<div class="tbl spot"><table><thead><tr>${sample.cols.map(c => `<th data-col="${esc(c)}">${esc(c)}</th>`).join('')}</tr></thead><tbody>${sample.rows.map((r, i) => `<tr>${r.map((v, j) => `<td data-k="${i}|${esc(sample.cols[j])}">${cell(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
      <div class="row"><span class="spot-tip hint">点击你觉得有问题的格子（也可以点表头）。</span></div>
      <div class="row"><span>已找到 <b class="cnt num">0</b> / ${total}</span><button class="all">全部揭晓</button></div>`;
    const tip = $('.spot-tip', el), cnt = $('.cnt', el);
    const hit = (td, key, f) => {
      td.classList.add(f[3] === 'ok' ? 'found-ok' : 'found');
      tip.className = 'spot-tip'; tip.innerHTML = `<b>${f[3] === 'ok' ? '陷阱题：' : '发现问题：'}</b>${esc(f[2] ?? f[1])}`;
      found.add(key); cnt.textContent = found.size;
    };
    $('table', el).onclick = e => {
      const td = e.target.closest('td,th'); if (!td) return;
      if (td.tagName === 'TH') {
        const f = C.get(td.dataset.col);
        if (f) { td.classList.add('found'); tip.className = 'spot-tip'; tip.innerHTML = `<b>整列有问题：</b>${esc(f[1])}`; found.add('col|' + f[0]); cnt.textContent = found.size; }
        else { tip.className = 'spot-tip hint'; tip.textContent = '这一列本身没问题，看看具体的格子。'; }
        return;
      }
      const f = F.get(td.dataset.k);
      if (f) hit(td, td.dataset.k, f);
      else { td.classList.add('miss'); tip.className = 'spot-tip hint'; tip.textContent = '这一格看起来正常。'; }
    };
    $('.all', el).onclick = () => {
      F.forEach((f, k) => { const td = $(`td[data-k="${CSS.escape(k)}"]`, el); if (td) td.classList.add(f[3] === 'ok' ? 'found-ok' : 'found'); });
      C.forEach((f, k) => $(`th[data-col="${CSS.escape(k)}"]`, el)?.classList.add('found'));
      tip.className = 'spot-tip'; tip.innerHTML = [...F.values()].map(f => `· <b>${esc(f[1])}</b>：${esc(f[2])}`).concat([...C.values()].map(f => `· <b>${esc(f[0])} 整列</b>：${esc(f[1])}`)).join('<br>');
    };
  }

  /* ---------- 表格 ---------- */
  function table(el, cols, rows, { right = [], fmts = {} } = {}) {
    el.innerHTML = `<div class="tbl"><table><thead><tr>${cols.map((c, j) => `<th class="${right.includes(j) ? 'r' : ''}">${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((v, j) => `<td class="${right.includes(j) ? 'r' : ''}">${fmts[j] ? fmt(v, fmts[j]) : cell(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  /* ---------- 图表：条形图（HTML） ---------- */
  function bars(el, items, { max, f = 'raw' } = {}) {
    const m = max ?? Math.max(...items.map(i => Math.abs(i.value)), 1e-9);
    el.innerHTML = `<div class="bars">${items.map(i => `<div class="bar ${i.cls || ''}"><span class="lbl" title="${esc(i.label)}">${esc(i.label)}</span><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(100, (i.value / m) * 100))}%"></span></span><span class="val">${i.text ?? fmt(i.value, f)}</span></div>`).join('')}</div>`;
  }

  /* ---------- 图表：SVG 坐标系 ---------- */
  function ticks(lo, hi, n = 5) {
    if (!(hi > lo)) hi = lo + 1;
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => s >= raw);
    const out = [];
    for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }
  function frame({ w = 640, h = 280, x, y, xfmt = v => v, yfmt = v => v, xlabel = '', ylabel = '', xt, yt }) {
    const m = { l: 56, r: 16, t: 14, b: xlabel ? 44 : 28 };
    const sx = v => m.l + ((v - x[0]) / (x[1] - x[0])) * (w - m.l - m.r);
    const sy = v => h - m.b - ((v - y[0]) / (y[1] - y[0])) * (h - m.t - m.b);
    let g = '';
    for (const t of yt ?? ticks(y[0], y[1])) g += `<line class="gridl" x1="${m.l}" x2="${w - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${yfmt(t)}</text>`;
    for (const t of xt ?? ticks(x[0], x[1], 6)) g += `<text x="${sx(t)}" y="${h - m.b + 16}" text-anchor="middle">${xfmt(t)}</text>`;
    g += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>`;
    if (xlabel) g += `<text x="${(m.l + w - m.r) / 2}" y="${h - 6}" text-anchor="middle">${esc(xlabel)}</text>`;
    if (ylabel) g += `<text x="${m.l}" y="${m.t - 2}" text-anchor="start" class="lbl-strong">${esc(ylabel)}</text>`;
    return { sx, sy, w, h, m, g };
  }
  const svg = (f, body, label) => `<svg class="chart" viewBox="0 0 ${f.w} ${f.h}" role="img" aria-label="${esc(label || '')}">${f.g}${body}</svg>`;
  const legend = series => `<div class="legend-row">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div>`;
  const extent = (vals, pad = 0.06) => {
    const v = vals.filter(Number.isFinite); let lo = Math.min(...v), hi = Math.max(...v);
    const p = (hi - lo || Math.abs(hi) || 1) * pad; return [lo - p, hi + p];
  };

  function lines(el, { x, series, ydom, xdom, marker, area, label, ...o }) {
    const yd = ydom ?? extent(series.flatMap(s => s.y));
    const f = frame({ x: xdom ?? [Math.min(...x), Math.max(...x)], y: yd, ...o });
    let body = '';
    for (const s of series) {
      const d = x.map((xv, i) => (s.y[i] == null ? '' : `${i ? 'L' : 'M'}${f.sx(xv).toFixed(1)},${f.sy(s.y[i]).toFixed(1)}`)).join('');
      if (area) body += `<path d="${d}L${f.sx(x[x.length - 1])},${f.sy(yd[0])}L${f.sx(x[0])},${f.sy(yd[0])}Z" style="fill:${s.color};opacity:.12"/>`;
      body += `<path d="${d}" style="fill:none;stroke:${s.color};stroke-width:${s.width || 2.2};${s.dash ? 'stroke-dasharray:5 4' : ''}"/>`;
      if (s.dots) body += x.map((xv, i) => `<circle cx="${f.sx(xv)}" cy="${f.sy(s.y[i])}" r="3" style="fill:${s.color}"/>`).join('');
    }
    if (marker) {
      const mx = f.sx(marker.x);
      body += `<line x1="${mx}" x2="${mx}" y1="${f.m.t}" y2="${f.h - f.m.b}" style="stroke:var(--ink);stroke-width:1;stroke-dasharray:3 3"/>`;
      (marker.points || []).forEach(p => (body += `<circle cx="${mx}" cy="${f.sy(p.y)}" r="5" style="fill:${p.color};stroke:var(--sheet);stroke-width:2"/>`));
      if (marker.text) body += `<text x="${Math.min(mx + 6, f.w - f.m.r - 4)}" y="${f.m.t + 12}" text-anchor="${mx > f.w * 0.7 ? 'end' : 'start'}" class="lbl-strong">${esc(marker.text)}</text>`;
    }
    el.innerHTML = svg(f, body, label) + (series.length > 1 ? legend(series) : '');
  }

  function scatter(el, { pts, colors, xdom, ydom, diag, r = 3, label, names, ...o }) {
    const xd = xdom ?? extent(pts.map(p => p[0])), yd = ydom ?? extent(pts.map(p => p[1]));
    const f = frame({ x: xd, y: yd, ...o });
    const lo = Math.max(xd[0], yd[0]), hi = Math.min(xd[1], yd[1]);
    let body = diag ? `<line x1="${f.sx(lo)}" y1="${f.sy(lo)}" x2="${f.sx(hi)}" y2="${f.sy(hi)}" style="stroke:var(--ink);stroke-dasharray:4 4;opacity:.6"/>` : '';
    body += pts.map(p => `<circle cx="${f.sx(p[0]).toFixed(1)}" cy="${f.sy(p[1]).toFixed(1)}" r="${r}" style="fill:${colors[p[2] || 0]};opacity:.7"/>`).join('');
    el.innerHTML = svg(f, body, label) + (names ? legend(names.map((n, i) => ({ name: n, color: colors[i] }))) : '');
  }

  function hist(el, { counts, edges }, { color = 'var(--accent)', xfmt = v => v, label, ...o } = {}) {
    const f = frame({ x: [edges[0], edges[edges.length - 1]], y: [0, Math.max(...counts) * 1.08], xfmt, ...o });
    const body = counts.map((c, i) => {
      const x0 = f.sx(edges[i]) + 0.5, x1 = f.sx(edges[i + 1]) - 0.5;
      return `<rect x="${x0}" y="${f.sy(c)}" width="${Math.max(x1 - x0, 0.5)}" height="${f.sy(0) - f.sy(c)}" style="fill:${color}"/>`;
    }).join('');
    el.innerHTML = svg(f, body, label);
  }

  /* ---------- 热力表：调参网格 ---------- */
  function heat(el, { rows, cols, rowName, colName, values, f = 'f3', better = 'max', onPick }) {
    const flat = values.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
    const best = better === 'max' ? hi : lo;
    const shade = v => { const t = (better === 'max' ? v - lo : hi - v) / (hi - lo || 1); return `color-mix(in srgb, var(--accent) ${Math.round(8 + t * 62)}%, var(--sheet))`; };
    el.innerHTML = `<div class="tbl heat"><table><thead><tr><th>${esc(rowName)} ↓ / ${esc(colName)} →</th>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map((r, i) => `<tr><th>${r}</th>${values[i].map((v, j) => `<td data-i="${i}" data-j="${j}" class="${v === best ? 'best' : ''}" style="background:${shade(v)}">${fmt(v, f)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    if (onPick) $('table', el).onclick = e => {
      const td = e.target.closest('td'); if (!td) return;
      $$('td.sel', el).forEach(t => t.classList.remove('sel')); td.classList.add('sel');
      onPick(+td.dataset.i, +td.dataset.j);
    };
  }

  /* ---------- 导轨与进度 ---------- */
  function rail() {
    const box = $('#rail'); if (!box) return;
    const steps = $$('.step');
    box.innerHTML = `<ol>${steps.map(s => `<li data-for="${s.id}"><a href="#${s.id}"><span class="joint">${s.dataset.step}</span><span>${esc(s.dataset.title)}</span>${$('.leak', s) ? '<i class="drip" title="本步有泄漏陷阱"></i>' : ''}</a></li>`).join('')}</ol><div class="legend"><span><i></i>本步有泄漏陷阱</span><span>实心节点 = 测验已完成</span></div>`;
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) $$('li', box).forEach(li => li.classList.toggle('here', li.dataset.for === e.target.id));
    }), { rootMargin: '-35% 0px -60% 0px' });
    steps.forEach(s => io.observe(s));
  }
  function refresh() {
    $$('.step').forEach(s => {
      const qs = $$('.quiz', s), done = qs.length > 0 && qs.every(q => q.dataset.id in prog.quiz);
      $(`#rail li[data-for="${s.id}"]`)?.classList.toggle('done', done);
    });
    $$('.leak[data-id]').forEach(l => { const s = $('.seen', l); if (s) s.textContent = leaks[l.dataset.id] ? '已排查' : '待排查：点开对比'; });
    const m = $('.top .meter');
    if (m && page !== 'home') {
      const steps = $$('.step'), done = steps.filter(s => $(`#rail li[data-for="${s.id}"]`)?.classList.contains('done')).length;
      const lk = $$('.leak[data-id]'), seen = lk.filter(l => leaks[l.dataset.id]).length;
      m.textContent = `步骤 ${done}/${steps.length} · 漏点 ${seen}/${lk.length}`;
    }
  }

  /* ---------- 每一步末尾附上 scikit-learn 官方文档 ---------- */
  const SK = 'https://scikit-learn.org/stable/';
  const DOCS = {
    0: [['选择合适的估计器', 'machine_learning_map.html']],
    1: [['入门 Getting Started', 'getting_started.html']],
    2: [['常见陷阱与最佳实践', 'common_pitfalls.html']],
    3: [['交叉验证与数据切分', 'modules/cross_validation.html']],
    4: [['Pipeline 与 ColumnTransformer', 'modules/compose.html'], ['数据预处理', 'modules/preprocessing.html']],
    5: [['Dummy 基线', 'modules/model_evaluation.html#dummy-estimators'], ['验证曲线', 'modules/learning_curve.html']],
    6: [['超参数搜索', 'modules/grid_search.html'], ['评估指标', 'modules/model_evaluation.html']],
    7: [['置换重要性', 'modules/permutation_importance.html'], ['模型持久化', 'model_persistence.html']],
  };
  const EXTRA = {
    reg: { 3: [['TimeSeriesSplit', 'modules/cross_validation.html#time-series-split']], 4: [['目标变换', 'modules/compose.html#transforming-target-in-regression'], ['TargetEncoder', 'modules/preprocessing.html#target-encoder']] },
    clf: { 7: [['调整分类阈值', 'modules/classification_threshold.html']] },
    clu: { 3: [['聚类总览', 'modules/clustering.html']], 5: [['K-means', 'modules/clustering.html#k-means'], ['聚类评估', 'modules/clustering.html#clustering-performance-evaluation']], 6: [['聚类评估', 'modules/clustering.html#clustering-performance-evaluation']] },
  };
  function docs() {
    $$('.step').forEach(s => {
      const n = +s.dataset.step, list = (page === 'clu' && n === 3 ? [] : DOCS[n] || []).concat(EXTRA[page]?.[n] || []);
      if (list.length) s.insertAdjacentHTML('beforeend', `<p class="hint">官方文档：${list.map(([t, u]) => `<a href="${SK + u}" target="_blank" rel="noopener">${esc(t)}</a>`).join(' · ')}</p>`);
    });
  }

  function init() {
    bind(); code(); tabs(); quizzes(); guesses(); orders(); classify(); reports(); docs(); rail();
    if (page !== 'home') { prog.total = $$('.quiz').length; store.set(page, prog); }
    refresh();
  }

  return { $, $$, get, fmt, esc, bind, code, tabs, spot, table, bars, lines, scatter, hist, heat, init, store, leaks, ticks };
})();
