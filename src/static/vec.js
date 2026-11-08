/* vec.js — embedding + id review, self-contained.
 *
 * Declares one global, `Vec`. It renders its own DOM into the two mount
 * points you give it and talks only to /api/vec/*. No shared state with
 * app.js beyond the optional callbacks.
 *
 *   <div class="card console" id="run-embed"></div>     <!-- empty -->
 *   <div id="rev-div"></div>                            <!-- empty -->
 *
 *   Vec.mount({
 *       embed:  '#run-embed',
 *       review: '#rev-div',
 *       onLog:   (msg, lvl) => logsWrite(msg, lvl),   // optional
 *   });
 *
 *   Vec.refresh();        // re-read status + queue
 *   Vec.loaded();         // bool, last known
 */
const Vec = (() => {
    'use strict';

    const API = '/api/vec';
    const CSS = `
.vec-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.vec-note{font:12px ui-monospace,Consolas,monospace;color:#71717a}
.vec-grid{display:grid;grid-template-columns:320px 1fr 200px;gap:14px;align-items:start}
.vec-queue{border:1px solid #e3e3e7;border-radius:10px;overflow:auto;max-height:62vh}
.vec-queue div{padding:8px 12px;border-bottom:1px solid #f2f2f4;cursor:pointer}
.vec-queue div:hover{background:#f7faff}
.vec-queue div.vec-sel{background:#eef4ff}
.vec-detail{border:1px solid #e3e3e7;border-radius:10px;padding:14px;max-height:62vh;
            overflow:auto}
.vec-cand{border:1px solid #e3e3e7;border-radius:9px;padding:10px 12px;margin-bottom:9px}
.vec-cand.vec-best{border-color:#0b62d0}
.vec-pid{font:600 15px ui-monospace,Consolas,monospace;color:#0b62d0}
.vec-ev{color:#71717a;font-size:12px;margin-top:3px}
.vec-series{font:12px ui-monospace,Consolas,monospace;color:#71717a;margin-top:3px}
.vec-meta{border-collapse:collapse;margin:4px 0 12px}
.vec-meta th{text-align:left;color:#71717a;font-weight:400;padding:1px 14px 1px 0}
.vec-w{display:flex;flex-direction:column;gap:8px}
.vec-w .vec-row{gap:6px}
.vec-w label{width:4.5rem;font-size:12px;color:#444}
.vec-w input{width:56px;text-align:right;font:inherit;padding:4px 6px;
             border:1px solid #d9d9de;border-radius:6px}

.vec-wr{display:flex;flex-direction:column;gap:8px}
.vec-wr .vec-row{gap:6px}
.vec-wr label{width:4.5rem;font-size:12px;color:#444}
.vec-wr input{width:56px;text-align:right;font:inherit;padding:4px 6px;
             border:1px solid #d9d9de;border-radius:6px}

.toolPanel{display:flex;flex-direction:column;gap:8px}
.toolPanel > .vec-wr{border-top:1px solid #e3e3e7;padding-top:16px}
`;

    const WEIGHTS = ['sim', 'money', 'dept', 'year'];

    let opts = {}, state = {loaded: false}, cands = [], cur = null;
    let elEmbed, elQueue, elDetail, elWeights, elWrite, elInfo, elDept, btnMake, btnLoad;

    /* ----------------------------------------------------------- helpers -- */

    const el = (tag, cls, text) => {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined) n.textContent = text;
        return n;
    };

    function say(msg, level = 'write') {
        if (typeof opts.onLog === 'function') opts.onLog(msg, level);
        else console.log('[vec]', level, msg);
    }

    async function call(url, init) {
        const res = await fetch(url, init);
        let body = null;
        try { body = await res.json(); } catch { /* not json */ }
        if (!res.ok || !body || body.ok === false) {
            throw new Error((body && body.error) || `${url} returned ${res.status}`);
        }
        return body;
    }

    const post = (url, obj) => call(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(obj || {}),
    });

    /* ------------------------------------------------------------ build -- */

    function style() {
        if (document.getElementById('vec-style')) return;
        const s = el('style');
        s.id = 'vec-style';
        s.textContent = CSS;
        document.head.append(s);
    }

    function buildEmbed(root) {
        root.replaceChildren();
        const row = el('div', 'vec-row');

        btnMake = el('button', 'vec-btn vec-primary', 'Create Embedding');
        btnLoad = el('button', 'vec-btn', 'Load Embedding');
        elInfo  = el('span', 'vec-note', '');

        btnMake.addEventListener('click', () => doLoad(true));
        btnLoad.addEventListener('click', () => doLoad(false));

        row.append(btnMake, btnLoad, elInfo);
        root.append(row);
    }

    function buildReview(root) {
        root.replaceChildren();

        const bar = el('div', 'vec-row');
        bar.style.marginBottom = '12px';
        elDept = el('select');
        elDept.append(new Option('all departments', ''));
        elDept.addEventListener('change', queue);
        const refreshBtn = el('button', 'vec-btn', 'Refresh');
        refreshBtn.addEventListener('click', refresh);
        bar.append(elDept, refreshBtn);

        elQueue   = el('div', 'vec-queue');
        elDetail  = el('div', 'vec-detail');
        elWeights = el('div', 'vec-w');
        elWrite   = el('div', 'vec-wr');
        elDetail.append(el('span', 'vec-note', 'Load the store, then pick a row.'));

        for (const k of WEIGHTS) {
            const r = el('div', 'vec-row');
            const lab = el('label', null, k);
            const inp = el('input');
            inp.type = 'number'; inp.min = 0; inp.max = 1; inp.step = 0.05;
            inp.id = `vec-w-${k}`;
            lab.htmlFor = inp.id;
            r.append(lab, inp);
            elWeights.append(r);
        }
        const applyBtn = el('button', 'vec-btn', 'Apply weights');
        applyBtn.addEventListener('click', applyWeights);
        elWeights.append(applyBtn);

        const wrow = el('div','vec-row');
        const wlbl = el('label',null,'Threshold:');
        const winp = el('input','vec-wr-inp');
        winp.type = 'number'; winp.min = 0; winp.max = 1; winp.step = 0.05;

        wrow.append(wlbl,winp)

        const writeBtn = el('button','write-btn','Write IDs');
        writeBtn.addEventListener('click', writeIDs);
        elWrite.append(wrow);
        elWrite.append(writeBtn);

        const grid = el('div', 'vec-grid');
        const toolPanel = el('div', 'toolPanel');
        toolPanel.append(elWeights,elWrite);

        grid.append(elQueue, elDetail, toolPanel);
        root.append(bar, grid);

        /* one listener for the whole queue */
        elQueue.addEventListener('click', e => {
            const row = e.target.closest('div[data-i]');
            if (!row) return;
            for (const d of elQueue.children) d.classList.remove('vec-sel');
            row.classList.add('vec-sel');
            detail(Number(row.dataset.i));
        });

        /* one listener for every "write id" button */
        elDetail.addEventListener('click', e => {
            const b = e.target.closest('button[data-pid]');
            if (b) assign(b.dataset.pid);
        });
    }

    /* ------------------------------------------------------------- data -- */

    async function doLoad(embed) {
        btnMake.disabled = btnLoad.disabled = true;
        try {
            await post(`${API}/load`, {embed});
            say(embed ? 'embedding started' : 'loading cached vectors', 'write');
            setTimeout(refresh, 1500);          // the job runs in a thread
        } catch (err) {
            say(err.message, 'error');
            btnMake.disabled = btnLoad.disabled = false;
        }
    }

    async function refresh() {
        let d;
        try { d = await call(API); } catch (err) { say(err.message, 'error'); return; }

        state = d;
        const s = d.stats || {};
        elInfo.textContent = d.loaded
            ? `${s.rows} rows · ${s.anchored} with id · ${s.orphans} without`
              + (s.vectors ? ` · ${s.vectors} vectors` : ' · names-only scoring')
            : 'not loaded';
        btnMake.disabled = btnLoad.disabled = false;

        if (d.weights) {
            for (const k of WEIGHTS) {
                const i = document.getElementById(`vec-w-${k}`);
                if (i && document.activeElement !== i) i.value = d.weights[k] ?? 0;
            }
        }
        if (d.loaded) queue();
    }

    async function queue() {
        let d;
        try {
            d = await call(`${API}/queue?dept=${encodeURIComponent(elDept.value || '')}`);
        } catch (err) { say(err.message, 'error'); return; }

        const want = new Set(['', ...d.departments]);
        if (elDept.options.length !== want.size) {
            const keep = elDept.value;
            elDept.replaceChildren(new Option('all departments', ''),
                ...d.departments.filter(Boolean).map(x => new Option(x, x)));
            elDept.value = keep;
        }

        elQueue.replaceChildren(...d.items.map(it => {
            const n = el('div');
            n.dataset.i = it.i;
            n.append(el('div', null, `${it.name || '(no name)'}`));
            const sub = el('div', 'vec-ev',
                `${it.year} · ${it.department || '—'} · ${it.top || 'no candidate'}`
                + ` · margin ${it.margin.toFixed(3)}`);
            n.append(sub);
            if (cur === it.i) n.classList.add('vec-sel');
            return n;
        }));
        if (!d.items.length) elQueue.append(el('div', 'vec-note', 'Nothing to review.'));
    }

    async function detail(i) {
        cur = i;
        let d;
        try { d = await call(`${API}/row/${i}`); } catch (err) { say(err.message, 'error'); return; }

        cands = d.candidates;
        const r = d.row;
        elDetail.replaceChildren();

        const h = el('h3', null, r.project_name || '(no name)');
        h.style.cssText = 'margin:0 0 6px;font-size:15px';
        elDetail.append(h);

        const t = el('table', 'vec-meta');
        for (const k of ['cip_year', 'department', 'address_location',
                         'previous_appropriations', 'project_total']) {
            if (!(k in r)) continue;
            const tr = el('tr');
            tr.append(el('th', null, k), el('td', null, String(r[k] ?? '')));
            t.append(tr);
        }
        elDetail.append(t, el('div', 'vec-ev', `margin ${d.margin.toFixed(3)}`));

        if (!cands.length) {
            elDetail.append(el('div', 'vec-note', 'no candidate passed the filters'));
        }
        cands.forEach((c, n) => {
            const box = el('div', 'vec-cand' + (n === 0 ? ' vec-best' : ''));
            const head = el('div', 'vec-row');
            head.append(el('span', 'vec-pid', c.pid),
                        el('span', 'vec-note', `score ${c.score.toFixed(3)}`));
            box.append(head, el('div', null, c.best_name),
                el('div', 'vec-ev', `sim ${c.sim.toFixed(3)} · money ${c.money.toFixed(2)}`
                    + ` · ${c.same_dept ? 'same dept' : 'DIFFERENT DEPT'}`),
                el('div', 'vec-series',
                    c.series.map(([y, v]) => `${y}: ${v.toLocaleString()}`).join('   ')));
            const b = el('button', 'vec-btn', `Write id  ${n + 1}`);
            b.dataset.pid = c.pid;
            b.style.marginTop = '8px';
            box.append(b);
            elDetail.append(box);
        });

        const manual = el('div', 'vec-row');
        manual.style.marginTop = '10px';
        const inp = el('input');
        inp.type = 'text'; inp.placeholder = 'type an id'; inp.id = 'vec-manual';
        const go = el('button', 'vec-btn', 'Write');
        go.addEventListener('click', () => assign(inp.value.trim()));
        manual.append(inp, go);
        elDetail.append(manual);
    }

    async function assign(pid) {
        if (!pid || cur === null) return;
        try {
            const d = await post(`${API}/assign`, {i: cur, pid});
            say(`${d.file} → ${d.pid}   (${d.orphans} left)`, 'write');
            cur = null;
            await queue();
            elDetail.replaceChildren(el('span', 'vec-note', 'Pick the next row.'));
        } catch (err) { say(err.message, 'error'); }
    }

    async function applyWeights() {
        const body = {};
        for (const k of WEIGHTS) {
            const i = document.getElementById(`vec-w-${k}`);
            if (i && i.value !== '') body[k] = Number(i.value);
        }
        try {
            const d = await post(`${API}/weights`, body);
            say('weights ' + WEIGHTS.map(k => `${k} ${d.weights[k].toFixed(2)}`).join(' · '));
            await queue();
            if (cur !== null) await detail(cur);
        } catch (err) { say(err.message, 'error'); }
    }

        async function writeIDs() {
        const t = Number(document.getElementById('vec-threshold').value);
        if (!(t > 0 && t <= 1)) { say('threshold must be between 0 and 1', 'yield'); return; }

        const dept = elDept.value || '';
        const btn  = elWrite.querySelector('button');
        btn.disabled = true;
        try {
            /* preview first — this writes to CSVs, so never fire blind */
            const dry = await post(`${API}/auto`, {threshold: t, dept, dry_run: true});
            if (!dry.written) {
                say(`nothing scores ≥ ${t.toFixed(2)} (${dry.considered} considered)`, 'yield');
                return;
            }
            for (const w of dry.sample) {
                say(`  ${w.name} → ${w.pid}  (${w.score.toFixed(3)})`);
            }
            if (dry.written > dry.sample.length) {
                say(`  … and ${dry.written - dry.sample.length} more`);
            }
            const where = dept ? ` in ${dept}` : '';
            if (!confirm(`Write ${dry.written} ids${where} at threshold ${t.toFixed(2)}?\n`
                       + `${dry.considered - dry.written} rows will be left for manual review.`)) {
                say('cancelled', 'yield');
                return;
            }

            const d = await post(`${API}/auto`, {threshold: t, dept});
            say(`wrote ${d.written} ids across ${d.files.length} file(s)`
                + ` — ${d.orphans} orphans left`, 'write');
            cur = null;
            elDetail.replaceChildren(el('span', 'vec-note', 'Pick a row.'));
            await queue();
        } catch (err) {
            say(err.message, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    /* keyboard: 1-6 writes the nth candidate */
    document.addEventListener('keydown', e => {
        if (!elDetail || !elDetail.isConnected) return;
        if (document.activeElement && /input|select|textarea/i
                .test(document.activeElement.tagName)) return;
        const n = Number(e.key);
        if (n >= 1 && n <= 6 && cands[n - 1]) assign(cands[n - 1].pid);
    });

    /* ------------------------------------------------------------ mount -- */

    function mount(mounts, options = {}) {
        opts = options;
        style();
        const pick = sel => typeof sel === 'string' ? document.querySelector(sel) : sel;

        const e = pick(mounts.embed), r = pick(mounts.review);
        if (!e || !r) {
            console.error('[vec] missing mount point', mounts);
            return api;
        }
        buildEmbed(e);
        buildReview(r);
        refresh();
        return api;
    }

    const api = {mount, refresh, loaded: () => !!state.loaded};
    return api;
})();