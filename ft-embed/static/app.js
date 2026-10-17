const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ---------------------------------------------------------------- tabs -- */
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
    document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('section').forEach(s =>
        s.classList.toggle('on', s.id === b.dataset.tab));
    if (b.dataset.tab === 'review') refreshQueue();
});

/* ------------------------------------------------------------- polling -- */
let lastRunning = false;
async function poll() {
    const d = await (await fetch('/api/status')).json();
    $('#jobstate').textContent = d.running ? `running: ${d.stage}` : 'idle';
    const text = (d.log || []).join('\n');
    for (const id of ['#log', '#log2']) {
        const el = $(id);
        if (el.textContent !== text) { el.textContent = text; el.scrollTop = el.scrollHeight; }
    }
    for (const k of ['csv_dir', 'out_dir', 'model_dir'])
        $('#p_' + k).textContent = d.config[k];
    $('#pairDir').textContent = d.config.csv_dir;
    $('#storeInfo').textContent = d.store
        ? `${d.store.rows} rows · ${d.store.anchored} with id · ${d.store.orphans} without`
          + (d.store.vectors ? ' · vectors' : ' · names only')
        : 'not loaded';
    document.querySelectorAll('button.act').forEach(b => b.disabled = d.running);
    if (lastRunning && !d.running) refreshQueue();     // a finished job may add rows
    lastRunning = d.running;
}
setInterval(poll, 1000); poll();

/* -------------------------------------------------------- folder picker -- */
let pickTarget = null, pickPath = null;
async function openPicker(target) {
    pickTarget = target;
    $('#pickerWrap').classList.add('on');
    const cur = $('#p_' + target).textContent;
    await listDir(cur && cur !== '—' ? cur : '');
}
function closePicker() { $('#pickerWrap').classList.remove('on'); }

async function listDir(path) {
    const d = await (await fetch('/api/browse?path=' + encodeURIComponent(path))).json();
    if (!d.ok) { alert(d.error); return; }
    pickPath = d.path;
    $('#pickPath').textContent = `${d.path}   (${d.csv_count} csv files here)`;
    $('#pickManual').value = d.path;

    // Paths are put in data-path rather than interpolated into an onclick
    // string: a Windows path contains backslashes, and "C:\Users" inside a JS
    // string literal is read as escape sequences (\U, \v) rather than text.
    const sep = d.path.includes('\\') ? '\\' : '/';
    const mk = (label, target) => {
        const el = document.createElement('div');
        el.textContent = '\u{1F4C1} ' + label;
        el.dataset.path = target;
        return el.outerHTML;
    };
    $('#dirs').innerHTML = mk('..', d.parent) +
        d.dirs.map(x => mk(x, d.path + sep + x)).join('');
}

// one listener for the whole list, so nothing has to be escaped into markup
$('#dirs').addEventListener('click', e => {
    const el = e.target.closest('div[data-path]');
    if (el) listDir(el.dataset.path);
});

async function choose() {
    const path = $('#pickManual').value || pickPath;
    await fetch('/api/config', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({[pickTarget]: path})});
    closePicker(); poll();
}

/* --------------------------------------------------------------- setup -- */
const post = (url, body) => fetch(url, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});

$('#btnLoad').onclick  = () => post('/api/vec/load', {embed: true});
$('#btnReuse').onclick = () => post('/api/vec/load', {embed: false});

$('#btnPair').onclick = () => {
    const steps = [];
    if ($('#s_pull').checked)    steps.push('pull');
    if ($('#s_section').checked) steps.push('section');
    if ($('#s_run').checked)     steps.push('run');
    if (!steps.length) return alert('pick at least one step');
    post('/api/pair', {steps});
};

/* -------------------------------------------------------------- review -- */
let items = [], cur = null, cands = [];

$('#btnRefresh').onclick = refreshQueue;
$('#dept').onchange = refreshQueue;

async function refreshQueue() {
    const d = await (await fetch('/api/vec/queue?dept=' +
                     encodeURIComponent($('#dept').value))).json();
    items = d.items || [];
    if (d.departments && $('#dept').options.length <= 1)
        $('#dept').innerHTML = '<option value="">all</option>' +
            d.departments.map(x => `<option>${esc(x)}</option>`).join('');
    $('#queueInfo').textContent = d.error ? d.error : `${d.remaining} rows without an id`;
    $('#queue').innerHTML = items.map((it, k) => `
        <div data-k="${k}" onclick="openRow(${k})">
          <b>${esc(it.name).slice(0, 40)}</b><br>
          <span class="ev">${it.year} · ${esc(it.department).slice(0, 22)}
          ${it.top ? '· ' + esc(it.top) + ' ' + it.score.toFixed(2) : '· no candidate'}</span>
        </div>`).join('') || '<div class="ev" style="padding:12px">queue empty</div>';
    if (items.length) openRow(0); else $('#detail').innerHTML =
        '<span class="path">Nothing to review.</span>';
}

async function openRow(k) {
    cur = items[k];
    document.querySelectorAll('#queue div').forEach(e =>
        e.classList.toggle('sel', +e.dataset.k === k));
    const d = await (await fetch('/api/vec/row/' + cur.i)).json();
    cands = d.candidates;
    const r = d.row;
    const money = ['previous_appropriations', 'project_total', 'spent', 'remaining']
        .filter(c => c in r).map(c => `<tr><th>${c}</th><td>${esc(r[c])}</td></tr>`).join('');
    $('#detail').innerHTML = `
      <h2 style="margin:0 0 6px;font-size:16px">${esc(r.project_name)}</h2>
      <table class="meta">
        <tr><th>year</th><td>${esc(r.cip_year)}</td></tr>
        <tr><th>department</th><td>${esc(r.department || '')}</td></tr>
        <tr><th>location</th><td>${esc(r.address_location || '')}</td></tr>${money}</table>
      <div class="ev" style="margin-bottom:10px">margin ${d.margin.toFixed(3)}</div>
      ${cands.map((c, i) => `
        <div class="cand ${i === 0 ? 'best' : ''}">
          <span class="pid">${esc(c.pid)}</span>
          <span style="float:right">score <b>${c.score.toFixed(3)}</b></span><br>
          ${esc(c.best_name)}
          <div class="ev">sim ${c.sim.toFixed(3)} · money ${c.money.toFixed(2)}
            · ${c.same_dept ? 'same dept' : 'DIFFERENT DEPT'}</div>
          <div class="series">${c.series.map(([y, t]) => y + ': ' + t.toLocaleString()).join('   ')}</div>
          <button class="act" style="margin-top:8px"
            onclick="assign('${esc(c.pid)}')">Write id <kbd>${i + 1}</kbd></button>
        </div>`).join('') || '<div class="ev">no candidate passed the filters</div>'}
      <div style="margin-top:10px"><input type="text" id="manualPid" placeholder="type an id">
        <button class="act" onclick="assign($('#manualPid').value)">Write</button></div>`;
}

async function assign(pid) {
    if (!pid) return;
    const d = await (await post('/api/vec/assign', {i: cur.i, pid})).json();
    if (d.ok) refreshQueue();
}

document.addEventListener('keydown', e => {
    if (!$('#review').classList.contains('on')) return;
    if (['INPUT', 'SELECT'].includes(e.target.tagName)) return;
    if (e.key >= '1' && e.key <= '6' && cands[+e.key - 1]) assign(cands[+e.key - 1].pid);
});
