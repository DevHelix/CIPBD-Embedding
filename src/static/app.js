Picker.mount('#folder-picker', { 
    onError: msg => logsWrite(msg, 'error'),
    onChange: paths => logsWrite(`csv_dir → ${paths.csv_dir}`, 'write') });

Vec.mount({embed: '#run-embed', review: '#rev-div'},
    {onLog: (msg, lvl) => logsWrite(msg, lvl)});

Clu.mount('#clu-div', {onLog: (msg, lvl) => logsWrite(msg, lvl)});

const logs     = document.getElementById("log");
const trainBtn = document.getElementById("trainBtn");

let shown = 0;

setInterval(async () => {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();

    if (d.log.length < shown) shown = 0;              // server restarted
    for (const [lvl, msg] of d.log.slice(shown)) logsWrite(msg, lvl);
    shown = d.log.length;

    trainBtn.disabled = d.running;
}, 1000);

for (child of document.getElementById("content").children){
    if(!child.classList.contains("active")){
        child.style.display = "none";
    }
}

function logsWrite(txt, t="write"){
    const row = document.createElement("div");
    row.className = `log log--${t}`;
    row.textContent = txt;
    logs.append(row);
    logs.scrollTop = logs.scrollHeight;        // keep the newest visible
    while (logs.childElementCount > 400) logs.firstElementChild.remove();
}

function topbarClick(element){
    for (const child of document.getElementById("content").children){
        child.classList.remove("active");
        child.style.display = "none";
    }
    element.classList.add("active");
    element.style.display = "block";
}

for (const child of document.getElementById("topbar").children){
    child.addEventListener("click",function(){
        var correspondingElem = document.getElementById(`${child.id}Cont`);
        if (!correspondingElem) {
            return;
        }
        topbarClick(correspondingElem);
    });
}

const post = (url, body) => fetch(url, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});

trainBtn.onclick = async e => {
    e.preventDefault();
    trainBtn.disabled = true;
    try {
        const body = {
            diagnosis:   document.getElementById('writeDiagnosis').checked,
            identity_col: 'project_id',
            epochs:       3,
        };
        const res  = await fetch('/train/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => null);
        if (!res.ok || !data || !data.ok) {
            logsWrite((data && data.error) || `server returned ${res.status}`, 'yield');
            return;
        }
        logsWrite(`training started on ${data.files} files`, 'write');
    } catch (err) {
        logsWrite(err.message, 'error');
    } finally {
        trainBtn.disabled = false;
    }
};

