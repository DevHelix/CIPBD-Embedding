/* picker.js — self-contained folder picker.
 *
 * Declares exactly one global, `Picker`, and touches no element outside the
 * root you give it plus the modal it creates. No $, no logsWrite, no shared
 * state with the rest of your app.
 *
 *   <div class="card" id="folder-picker">
 *     <div class="row pick" data-key="csv_dir">
 *       <button class="picker">Choose a CSV Path</button>
 *       <label>—</label>
 *     </div>
 *   </div>
 *
 *   Picker.mount('#folder-picker');                      // that's it
 *   Picker.mount('#folder-picker', {
 *       onChange: paths => console.log(paths),           // after every pick
 *       onError:  msg   => logsWrite(msg, 'error'),      // optional
 *   });
 *
 *   await Picker.refresh();      // re-read from the server
 *   Picker.paths();              // {csv_dir: '...'}  last known, sync
 */
const Picker = (() => {
    'use strict';

    const API = '/api/paths';
    const CSS = `
.pk-wrap{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;
         align-items:center;justify-content:center;z-index:9999}
.pk-wrap.pk-on{display:flex}
.pk-box{background:#fff;color:#1d1d1f;border-radius:12px;width:min(620px,92vw);
        max-height:76vh;display:flex;flex-direction:column;overflow:hidden;
        font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.pk-head,.pk-foot{padding:12px 18px;flex:none}
.pk-head{border-bottom:1px solid #e3e3e7;font:12px ui-monospace,Consolas,monospace;
         word-break:break-all}
.pk-foot{border-top:1px solid #e3e3e7;display:flex;gap:10px;align-items:center}
.pk-foot input{flex:1;font:inherit;padding:7px 10px;border:1px solid #e3e3e7;
               border-radius:7px;min-width:0}
.pk-foot button{font:inherit;padding:7px 16px;border:1px solid #e3e3e7;
                background:#fafafa;border-radius:7px;cursor:pointer}
.pk-foot button.pk-primary{background:#0b62d0;color:#fff;border-color:#0b62d0}
.pk-list{overflow:auto;flex:1;padding:6px 0}
.pk-list div{padding:6px 18px;cursor:pointer}
.pk-list div:hover{background:#f2f6ff}
.pk-err{color:#c0392b;padding:8px 18px;font-size:13px}
`;

    let wrap, head, list, manual, target = null, here = null;
    let opts = {}, known = {};

    /* ------------------------------------------------------------- modal -- */

    function build() {
        if (wrap) return;

        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.append(style);

        wrap = document.createElement('div');
        wrap.className = 'pk-wrap';

        const box = document.createElement('div');
        box.className = 'pk-box';

        head = document.createElement('div');
        head.className = 'pk-head';

        list = document.createElement('div');
        list.className = 'pk-list';

        const foot = document.createElement('div');
        foot.className = 'pk-foot';

        manual = document.createElement('input');
        manual.type = 'text';
        manual.placeholder = 'or type a path';

        const cancel = document.createElement('button');
        cancel.textContent = 'Cancel';
        cancel.addEventListener('click', close);

        const use = document.createElement('button');
        use.className = 'pk-primary';
        use.textContent = 'Use this folder';
        use.addEventListener('click', commit);

        foot.append(manual, cancel, use);
        box.append(head, list, foot);
        wrap.append(box);
        document.body.append(wrap);

        /* one delegated listener for every folder row */
        list.addEventListener('click', e => {
            const el = e.target.closest('div[data-path]');
            if (el) listDir(el.dataset.path);
        });

        /* click the backdrop or press Escape to dismiss */
        wrap.addEventListener('click', e => { if (e.target === wrap) close(); });
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && wrap.classList.contains('pk-on')) close();
            if (e.key === 'Enter'  && wrap.classList.contains('pk-on')) commit();
        });
    }

    function close() { wrap.classList.remove('pk-on'); }

    function fail(msg) {
        if (typeof opts.onError === 'function') opts.onError(msg);
        else console.error('[picker]', msg);
    }

    /* -------------------------------------------------------------- data -- */

    async function json(url, init) {
        const res = await fetch(url, init);
        if (!res.ok) throw new Error(`${url} returned ${res.status}`);
        return res.json();
    }

    async function listDir(path) {
        let d;
        try {
            d = await json(`${API}/browse?path=${encodeURIComponent(path || '')}`);
        } catch (err) { fail(err.message); return; }
        if (!d.ok) { fail(d.error); return; }

        here = d.path;
        head.textContent = `${d.path}   (${d.csv_count} csv here)`;
        manual.value = d.path;

        /* Paths live in dataset, never in markup: a Windows path contains
           backslashes, and "C:\\Users" inside a string literal would be read
           as escape sequences rather than text. */
        const sep = d.path.includes('\\') ? '\\' : '/';
        const row = (label, full) => {
            const el = document.createElement('div');
            el.textContent = '\u{1F4C1} ' + label;
            el.dataset.path = full;
            return el;
        };
        list.replaceChildren(
            row('..', d.parent),
            ...d.dirs.map(name => row(name, d.path + sep + name))
        );
        list.scrollTop = 0;
    }

    async function commit() {
        const path = manual.value.trim() || here;
        if (!path || !target) return;
        let d;
        try {
            d = await json(API, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[target]: path}),
            });
        } catch (err) { fail(err.message); return; }
        if (!d.ok) { fail(d.error); return; }

        known = d.paths;
        paint();
        close();
        if (typeof opts.onChange === 'function') opts.onChange({...known});
    }

    /* -------------------------------------------------------------- view -- */

    function paint() {
        if (!opts.root) return;
        for (const row of opts.root.querySelectorAll('.row.pick[data-key]')) {
            const label = row.querySelector('label');
            if (label) label.textContent = known[row.dataset.key] || '—';
        }
    }

    async function refresh() {
        try {
            const d = await json(API);
            known = d.paths || {};
            paint();
            return {...known};
        } catch (err) { fail(err.message); return {}; }
    }

    /* ------------------------------------------------------------- mount -- */

    function mount(rootSelector, options = {}) {
        const root = typeof rootSelector === 'string'
            ? document.querySelector(rootSelector) : rootSelector;
        if (!root) { console.error('[picker] no such root:', rootSelector); return; }

        opts = {...options, root};
        build();

        root.addEventListener('click', e => {
            const btn = e.target.closest('button.picker');
            if (!btn || !root.contains(btn)) return;
            const row = btn.closest('.row.pick[data-key]');
            if (!row) return;
            target = row.dataset.key;
            const cur = (row.querySelector('label')?.textContent || '').trim();
            wrap.classList.add('pk-on');
            listDir(cur && cur !== '—' && cur !== '...' ? cur : '');
            manual.focus();
        });

        refresh();
        return api;
    }

    const api = {mount, refresh, paths: () => ({...known})};
    return api;
})();
