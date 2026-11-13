/* clusters.js — inspect and edit clusters before writing them, self-contained.
 *
 * Declares one global, `Clu`. Renders into one mount point and talks only to
 * /api/vec/cluster*. Nothing is written to the CSVs until "Write" is pressed;
 * every edit changes a server-side draft that can be undone.
 *
 *   <div id="clu-div"></div>
 *   Clu.mount('#clu-div', {onLog: (msg, lvl) => logsWrite(msg, lvl)});
 *   Clu.refresh();
 *
 * Keys (while the tab is visible and no text box has focus):
 *   Ctrl/Cmd+Z undo · Ctrl/Cmd+Shift+Z or Ctrl+Y redo
 */
const Clu = (() => {
    'use strict';

    const API = '/api/vec/cluster';
    const NOISE = -1;
    const CSS = `
.clu-grid{display:grid;grid-template-columns:330px 1fr 230px;gap:14px;align-items:start}
.clu-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.clu-note{font:12px ui-monospace,Consolas,monospace;color:#71717a}
.clu-list{border:1px solid #e3e3e7;border-radius:10px;overflow:auto;max-height:64vh}
.clu-list > div{padding:7px 12px;border-bottom:1px solid #f2f2f4;cursor:pointer}
.clu-list > div:hover{background:#f7faff}
.clu-list > div.clu-sel{background:#eef4ff}
.clu-id{font:600 13px ui-monospace,Consolas,monospace;color:#0b62d0}
.clu-sub{color:#71717a;font-size:12px;margin-top:2px}
.clu-warn{color:#8a6d00}
.clu-detail{border:1px solid #e3e3e7;border-radius:10px;padding:14px;max-height:64vh;overflow:auto}
.clu-detail h3{margin:0 0 4px;font-size:15px}
.clu-table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
.clu-table th{text-align:left;color:#71717a;font-weight:400;border-bottom:1px solid #e3e3e7;
              padding:4px 6px;position:sticky;top:0;background:#fff}
.clu-table td{padding:4px 6px;border-bottom:1px solid #f2f2f4;vertical-align:top}
.clu-table td.clu-dupyr{background:#fff4d6;color:#8a6d00;font-weight:600}
.clu-table td.num{text-align:right;font-variant-numeric:tabular-nums}
.clu-fit-low{color:#c0392b}
.clu-badge{font:11px ui-monospace,Consolas,monospace;padding:1px 6px;border-radius:9px;
           background:#f2f2f4;color:#444}
.clu-badge.set{background:#e7f5ec;color:#1e7a43}
.clu-badge.cleared{background:#fdecea;color:#c0392b}
.clu-badge.protected{background:#fff4d6;color:#8a6d00}
.clu-bar{margin-top:10px;padding-top:10px;border-top:1px solid #e3e3e7}
.clu-panel{display:flex;flex-direction:column;gap:10px}
.clu-panel table{border-collapse:collapse;font-size:13px}
.clu-panel td{padding:2px 10px 2px 0}
.clu-in{width:90px;font:inherit;padding:4px 6px;border:1px solid #d9d9de;border-radius:6px}
.clu-wide{flex:1;min-width:120px;font:inherit;padding:4px 6px;border:1px solid #d9d9de;
          border-radius:6px}
.clu-find > div{display:flex;gap:8px;align-items:center;padding:4px 0;
                border-bottom:1px solid #f2f2f4;font-size:13px}
.clu-find > div span:first-child{flex:1}
.clu-link{color:#0b62d0;cursor:pointer;text-decoration:underline}
`;

    let opts = {}, view = null, cur = null, members = [];
    let elList, elDetail, elCounts, elFilter, elSort, btnFit, btnUndo, btnRedo, btnWrite, chkOver, rootEl;

    /* ----------------------------------------------------------- helpers -- */

    const el = (tag, cls, text) => {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined && text !== null) n.textContent = text;
        return n;
    };
    const btn = (text, fn, cls = 'vec-btn') => {
        const b = el('button', cls, text);
        b.addEventListener('click', fn);
        return b;
    };

    function say(msg, level = 'write') {
        if (typeof opts.onLog === 'function') opts.onLog(msg, level);
        else console.log('[clu]', level, msg);
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
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(obj || {}),
    });
    const over = () => (chkOver && chkOver.checked ? '?overwrite=1' : '');

    /** "cl0007" | "7" -> 7, "noise" -> -1, "new" | "" -> null, else NaN */
    function parseTarget(s) {
        s = String(s || '').trim().toLowerCase();
        if (s === '' || s === 'new') return null;
        if (s === 'noise') return NOISE;
        const m = s.match(/^(?:cl)?(\d+)$/);
        return m ? Number(m[1]) : NaN;
    }

    const money = v => (v === null || v === undefined) ? '' : Number(v).toLocaleString();

    /* ------------------------------------------------------------ build -- */

    function style() {
        if (document.getElementById('clu-style')) return;
        const s = el('style'); s.id = 'clu-style'; s.textContent = CSS;
        document.head.append(s);
    }

    function build(root) {
        root.replaceChildren();

        /* left: actions, filter, list */
        const left = el('div', 'clu-panel');
        btnFit  = btn('Run clustering', fit, 'vec-btn vec-primary');
        btnUndo = btn('Undo', () => step('undo'));
        btnRedo = btn('Redo', () => step('redo'));
        btnUndo.disabled = btnRedo.disabled = true;
        const top = el('div', 'clu-row'); top.append(btnFit, btnUndo, btnRedo);

        elFilter = el('input', 'clu-wide');
        elFilter.placeholder = 'filter: name, cl id, department';
        elFilter.addEventListener('input', renderList);
        elSort = el('select');
        for (const [v, t] of [['label', 'by number'], ['size', 'largest'],
                              ['dup', 'duplicate years'], ['changes', 'most changes']]) {
            elSort.append(new Option(t, v));
        }
        elSort.addEventListener('change', renderList);
        const fl = el('div', 'clu-row'); fl.append(elFilter, elSort);

        elList = el('div', 'clu-list');
        elList.addEventListener('click', e => {
            const d = e.target.closest('div[data-label]');
            if (d) open(Number(d.dataset.label));
        });
        const pAll = el('input', 'clu-in');
        pAll.type = 'number'; pAll.min = 0; pAll.max = 1; pAll.step = 0.05; pAll.value = 0.7;
        const pr = el('div', 'clu-row');
        pr.append(el('span', 'clu-note', 'every cluster: fit below'), pAll,
                  btn('Prune all', () => prune(null, pAll.value)));

        left.append(top, pr, fl, elList);

        /* middle: one cluster */
        elDetail = el('div', 'clu-detail');
        elDetail.append(el('span', 'clu-note', 'Load an embedding, then run clustering.'));

        /* right: write panel */
        const right = el('div', 'clu-panel');
        right.append(el('strong', null, 'Write to CSVs'));
        elCounts = el('div');
        const ol = el('label', 'clu-row');
        chkOver = el('input'); chkOver.type = 'checkbox';
        chkOver.addEventListener('change', async () => { await refresh(); if (cur !== null) await open(cur); });
        ol.append(chkOver, el('span', null, 'overwrite review marks'));
        btnWrite = btn('Write', write, 'write-btn');
        btnWrite.disabled = true;
        right.append(elCounts, ol, btnWrite,
            el('div', 'clu-note', 'Edits only change the draft. Nothing reaches '
                + 'the CSVs until you press Write.'));

        const grid = el('div', 'clu-grid');
        grid.append(left, elDetail, right);
        root.append(grid);
    }

    /* ------------------------------------------------------------- list -- */

    function renderList() {
        if (!view) { elList.replaceChildren(); return; }
        const q = elFilter.value.trim().toLowerCase();
        let cs = view.clusters.filter(c => !q
            || c.id.includes(q) || c.name.toLowerCase().includes(q)
            || c.departments.some(d => d.toLowerCase().includes(q)));
        const key = {label: c => c.label, size: c => -c.size,
                     dup: c => -c.dup_years, changes: c => -c.changes}[elSort.value];
        cs = cs.slice().sort((a, b) => key(a) - key(b) || a.label - b.label);

        const items = [];
        const noise = el('div'); noise.dataset.label = NOISE;
        noise.append(el('span', 'clu-id', 'noise'),
                     el('div', 'clu-sub', `${view.noise.size} rows in no cluster`));
        if (cur === NOISE) noise.classList.add('clu-sel');
        items.push(noise);

        for (const c of cs.slice(0, 1500)) {
            const d = el('div'); d.dataset.label = c.label;
            const head = el('div', 'clu-row');
            head.append(el('span', 'clu-id', c.id), el('span', 'clu-note', `${c.size} rows`));
            const sub = el('div', 'clu-sub',
                `${c.years[0]}–${c.years[1]} · ${c.departments.join(', ') || '—'}`);
            if (c.dup_years) sub.append(el('span', 'clu-warn', ` · ${c.dup_years} dup yr`));
            if (c.changes) sub.append(` · ${c.changes} to write`);
            d.append(head, el('div', null, c.name), sub);
            if (c.label === cur) d.classList.add('clu-sel');
            items.push(d);
        }
        if (cs.length > 1500) items.push(el('div', 'clu-note', `… ${cs.length - 1500} more, use the filter`));
        elList.replaceChildren(...items);
    }

    function renderCounts() {
        const n = view ? view.counts : {};
        const t = el('table');
        for (const [k, label] of [['set', 'set'], ['cleared', 'cleared'],
                                  ['protected', 'kept (review marks)'], ['unchanged', 'unchanged']]) {
            const tr = el('tr');
            tr.append(el('td', null, label), el('td', 'num', String(n[k] ?? 0)));
            t.append(tr);
        }
        const todo = (n.set || 0) + (n.cleared || 0);
        elCounts.replaceChildren(t);
        btnWrite.textContent = todo ? `Write ${todo} rows` : 'Nothing to write';
        btnWrite.disabled = !todo;
        btnUndo.disabled = !(view && view.can_undo);
        btnRedo.disabled = !(view && view.can_redo);
        btnUndo.title = view && view.undo_label ? `Undo ${view.undo_label} (Ctrl+Z)` : 'Nothing to undo';
        btnRedo.title = view && view.redo_label ? `Redo ${view.redo_label} (Ctrl+Shift+Z)` : 'Nothing to redo';
    }

    /* ----------------------------------------------------------- detail -- */

    async function open(label) {
        cur = label;
        let d;
        try { d = await call(`${API}/${label}${over()}`); }
        catch (err) { say(err.message, 'error'); return; }
        members = d.members;
        for (const n of elList.children) {
            n.classList.toggle('clu-sel', Number(n.dataset.label) === label);
        }
        renderDetail(d);
    }

    function renderDetail(d) {
        elDetail.replaceChildren();
        const isNoise = d.label === NOISE;
        const info = view && view.clusters.find(c => c.label === d.label);

        elDetail.append(el('h3', null, isNoise ? 'Noise' : `${d.id} — ${info ? info.name : ''}`));
        elDetail.append(el('div', 'clu-note', isNoise
            ? `${d.size} rows in no cluster. Select rows to put them in one.`
            : `${d.size} rows · sorted by fit (cosine to the cluster centre), worst first`
              + ' · highlighted years appear more than once in this cluster'));

        /* cluster actions */
        if (!isNoise) {
            const bar = el('div', 'clu-row clu-bar');
            const mergeIn = el('input', 'clu-in'); mergeIn.placeholder = 'cl id';
            bar.append(el('span', null, 'Merge this into'), mergeIn,
                btn('Merge', () => {
                    const t = parseTarget(mergeIn.value);
                    if (!(t >= 0)) return say('type the cluster to merge into, e.g. cl0012', 'yield');
                    edit({op: 'merge', sources: [d.label, t], target: t}, t);
                }),
                btn('Dissolve to noise', () => {
                    if (confirm(`Send all ${d.size} rows of ${d.id} to noise?`)) {
                        edit({op: 'dissolve', label: d.label}, NOISE);
                    }
                }));
            const pIn = el('input', 'clu-in');
            pIn.type = 'number'; pIn.min = 0; pIn.max = 1; pIn.step = 0.05; pIn.value = 0.7;
            const pbar = el('div', 'clu-row');
            pbar.style.marginTop = '8px';
            pbar.append(el('span', null, 'Remove members with fit below'), pIn,
                        btn('Prune', () => prune(d.label, pIn.value)));
            elDetail.append(bar, pbar);
        }

        /* member table */
        const t = el('table', 'clu-table');
        const head = el('tr');
        const all = el('input'); all.type = 'checkbox';
        all.addEventListener('change', () =>
            t.querySelectorAll('input[data-i]').forEach(c => { c.checked = all.checked; }));
        const th0 = el('th'); th0.append(all); head.append(th0);
        for (const h of ['fit', 'year', 'name', 'department', 'total', 'project_id', 'id_modified']) {
            head.append(el('th', null, h));
        }
        t.append(head);
        for (const m of members) {
            const tr = el('tr');
            const cb = el('input'); cb.type = 'checkbox'; cb.dataset.i = m.i;
            const td0 = el('td'); td0.append(cb); tr.append(td0);
            const fit = el('td', 'num' + (m.fit !== null && m.fit < 0.7 ? ' clu-fit-low' : ''),
                           m.fit === null ? '' : m.fit.toFixed(3));
            const mark = el('td');
            mark.append(el('span', 'clu-note', m.mark || '—'), ' ',
                        el('span', `clu-badge ${m.action}`, m.action));
            tr.append(fit, el('td', m.dup_year && !isNoise ? 'clu-dupyr' : null, m.year),
                      el('td', null, m.name),
                      el('td', null, m.department), el('td', 'num', money(m.total)),
                      el('td', 'clu-note', m.project_id || ''), mark);
            t.append(tr);
        }

        /* selection actions */
        const sel = () => [...t.querySelectorAll('input[data-i]:checked')].map(c => Number(c.dataset.i));
        const need = fn => () => { const r = sel(); if (!r.length) return say('select rows first', 'yield'); fn(r); };
        const sb = el('div', 'clu-row clu-bar');
        const moveIn = el('input', 'clu-in'); moveIn.placeholder = 'cl id';
        sb.append(el('span', null, 'Selected rows:'),
            btn('Move to', need(r => {
                const to = parseTarget(moveIn.value);
                if (Number.isNaN(to) || to === null) return say('type a cluster id, "noise" or "new"', 'yield');
                edit({op: 'move', rows: r, to}, d.label);
            })), moveIn,
            btn('New cluster', need(r => edit({op: 'move', rows: r, to: null}, null))));
        if (!isNoise) sb.append(btn('Remove to noise', need(r => edit({op: 'move', rows: r, to: NOISE}, d.label))));
        elDetail.append(sb);
        elDetail.append(t);

        /* add rows from anywhere */
        if (!isNoise) {
            const fb = el('div', 'clu-bar');
            const q = el('input', 'clu-wide'); q.placeholder = `find rows by name to add to ${d.id}`;
            const res = el('div', 'clu-find');
            const go = async () => {
                if (!q.value.trim()) return;
                let r;
                try { r = await call(`${API}/find?q=${encodeURIComponent(q.value)}`); }
                catch (err) { return say(err.message, 'error'); }
                res.replaceChildren(...r.rows.map(x => {
                    const row = el('div');
                    const where = el('span', x.label === d.label ? 'clu-note' : 'clu-link',
                                     x.id || 'noise');
                    if (x.label !== d.label) where.addEventListener('click', () => open(x.label));
                    row.append(el('span', null, `${x.year} · ${x.name} · ${x.department}`), where);
                    if (x.label !== d.label) {
                        row.append(btn('Add', () => edit({op: 'move', rows: [x.i], to: d.label}, d.label)));
                    }
                    return row;
                }));
                if (!r.rows.length) res.replaceChildren(el('span', 'clu-note', 'no matches'));
            };
            q.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
            const fr = el('div', 'clu-row'); fr.append(q, btn('Find', go));
            fb.append(fr, res);
            elDetail.append(fb);
        }
    }

    /* ------------------------------------------------------------ actions -- */

    /** Apply one edit, then show `show` (null = the label the server returns). */
    async function edit(body, show) {
        let r;
        try { r = await post(`${API}/edit`, body); }
        catch (err) { return say(err.message, 'error'); }
        await refresh();
        const next = show === null || show === undefined ? r.label : show;
        if (next !== undefined && next !== null) await open(next);
    }

    /** Send members below `threshold` fit to noise: one cluster, or all (label null). */
    async function prune(label, threshold) {
        const t = Number(threshold);
        if (!(t > 0 && t <= 1)) return say('threshold must be between 0 and 1', 'yield');
        const body = {op: 'prune', threshold: t, label};
        let dry;
        try { dry = await post(`${API}/edit`, {...body, dry_run: true}); }
        catch (err) { return say(err.message, 'error'); }
        const where = label === null ? `across ${dry.clusters} cluster(s)` : 'in this cluster';
        if (!dry.removed) return say(`no members with fit below ${t.toFixed(2)} ${where}`, 'yield');
        if (!confirm(`Send ${dry.removed} rows with fit below ${t.toFixed(2)} ${where} to noise?`)) return;
        try { await post(`${API}/edit`, body); }
        catch (err) { return say(err.message, 'error'); }
        say(`pruned ${dry.removed} rows below ${t.toFixed(2)} ${where}`);
        await refresh();
        if (cur !== null) await open(cur);
    }

    async function fit() {
        if (view && !confirm('Re-cluster from scratch? Your edits to the current draft are lost.')) return;
        btnFit.disabled = true;
        say('clustering…');
        try {
            await post(`${API}/fit`);
            cur = null;
            await refresh();
            elDetail.replaceChildren(el('span', 'clu-note', 'Pick a cluster.'));
            say(`${view.clusters.length} clusters · ${view.noise.size} noise`);
        } catch (err) { say(err.message, 'error'); }
        finally { btnFit.disabled = false; }
    }

    let stepping = false;

    /** op = 'undo' | 'redo'. Re-opens the current cluster, or noise if it is gone. */
    async function step(op) {
        if (stepping || !view || !view[`can_${op}`]) return;
        stepping = true;
        try {
            const r = await post(`${API}/edit`, {op});
            const what = r.undone || r.redone;
            if (what) say(`${op === 'undo' ? 'undid' : 'redid'}: ${what}`);
            await refresh();
            if (cur !== null) {
                const alive = cur === NOISE || view.clusters.some(c => c.label === cur);
                await open(alive ? cur : NOISE);
            }
        } catch (err) { say(err.message, 'error'); }
        finally { stepping = false; }
    }

    document.addEventListener('keydown', e => {
        if (!rootEl || !rootEl.isConnected || rootEl.offsetParent === null) return;  // tab hidden
        if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
        const a = document.activeElement;
        if (a && /^(input|select|textarea)$/i.test(a.tagName) && a.type !== 'checkbox') return;
        const k = e.key.toLowerCase();
        if (k === 'z' && !e.shiftKey) { e.preventDefault(); step('undo'); }
        else if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); step('redo'); }
    });

    async function write() {
        const n = view.counts;
        if (!confirm(`Write cluster ids to ${n.set + n.cleared} rows?\n`
                   + `${n.set} set · ${n.cleared} cleared · ${n.protected} review marks kept`)) return;
        btnWrite.disabled = true;
        try {
            const d = await post(API, {overwrite: chkOver.checked});
            say(`wrote ${d.written} cluster ids across ${d.files.length} file(s)`, 'write');
            for (const s of d.skipped) say(`skipped ${s.file} row ${s.row}: ${s.reason}`, 'yield');
            await refresh();
            if (cur !== null) await open(cur);
        } catch (err) { say(err.message, 'error'); renderCounts(); }
    }

    async function refresh() {
        try { view = await call(`${API}${over()}`); }
        catch { view = null; }            // no draft yet: not an error
        btnFit.textContent = view ? 'Re-cluster' : 'Run clustering';
        renderList();
        renderCounts();
    }

    /* ------------------------------------------------------------ mount -- */

    function mount(sel, options = {}) {
        opts = options;
        style();
        const root = typeof sel === 'string' ? document.querySelector(sel) : sel;
        if (!root) { console.error('[clu] missing mount point', sel); return api; }
        rootEl = root;
        build(root);
        refresh();
        return api;
    }

    const api = {mount, refresh};
    return api;
})();
