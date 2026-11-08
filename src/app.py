import sys
import pathlib
import os
from flask import Flask, jsonify, render_template, request
import paths
from paths import bp as paths_bp
import threading
import traceback
import vectors


app = Flask(__name__)
app.register_blueprint(paths_bp) # black box directory picker

app.register_blueprint(vectors.bp) # black box vector interface

ROOT = pathlib.Path(__file__).resolve().parent

JOB = {"running": False, "log": []}

def log(msg, level="write"):
    JOB["log"].append([level, str(msg)])
    del JOB["log"][:-400]

def run_job(fn, *a, **kw):
    def worker():
        JOB.update(running=True, stage=fn.__name__)
        try:
            fn(*a, **kw)
        except Exception as e:
            log(f"ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc(limit=3))
        finally:
            JOB.update(running=False, stage="")
    threading.Thread(target=worker, daemon=True).start()

vectors.configure(log=log, runner=run_job)

def check_file():
    return (ROOT.parent / "data" / "config.json").is_file()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify(JOB)

@app.route("/train/run", methods=["POST"])
def train_run():
    if not paths.ready():
        return jsonify({"ok": False, "error": "choose a CSV folder first"}), 400

    files = paths.csv_files()          # ['...\\2010.csv', '...\\2011.csv', ...]
    if not files:
        return jsonify({"ok": False, "error": f"no NNNN.csv in {paths.get('csv_dir')}"}), 400
    
    d = request.get_json(silent=True) or {}
    cfg = {
        "diagnosis":    bool(d.get("diagnosis", False)),
        "identity_col": str(d.get("identity_col") or "project_id"),
        "epochs":       max(1, min(int(d.get("epochs", 3)), 20)),
        "csv_dir":      paths.get("csv_dir"),
        "train_dir":    paths.get("out_dir"),
    }

    def work(cfg):
        from train import loss_pairing
        p = loss_pairing(csv_dir=cfg["csv_dir"], train_dir=cfg["train_dir"], anchorColumn=cfg["identity_col"])
        log(f"reading {len(files)} files, identity={cfg['identity_col']}")
        p.getData()
        p.getDiff()
        pairs = p.getPairs()
        log(f"{len(pairs)} training pairs")
        if cfg["diagnosis"]:
            p.writeDiagnosis()
        p.train(pairs, epochs=cfg["epochs"])

    run_job(work, cfg)
    return jsonify({"ok": True, "files": len(files), "config": cfg})

if __name__ == "__main__":
    print(app.url_map) 
    app.run(debug=True, use_reloader=False, threaded=True)
