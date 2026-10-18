"""Build blind listening pages: outputs/labeler/primary.html and second.html.

Blinding: clips are renamed to opaque keys, order is shuffled, and the page shows ONLY the prompt
text and the audio. No model name, no transcript, no detector output.

Usage:
  python scripts/04_build_labeler.py                  # all original clips + 20% second-listener subset
  python scripts/04_build_labeler.py --source regen   # unlabelled clips produced by 06_regenerate.py
Open the HTML file in Chrome from the outputs/labeler folder (audio files sit beside it).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.dataset import clip_key  # noqa: E402
from tts_verifier.io import DATA, LABELS, OUT, read_jsonl, read_labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["original", "regen"], default="original")
ap.add_argument("--second-fraction", type=float, default=0.2)
args = ap.parse_args()

prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
gens = read_jsonl(OUT / ("generations.jsonl" if args.source == "original" else "regen.jsonl"))
if args.source == "regen":   # only the clips the policy actually kept need human labels
    finals = {p["final_clip"] for p in read_jsonl(OUT / "regen_policy.jsonl")}
    gens = [g for g in gens if g["clip_id"] in finals]
already = read_labels(LABELS / "primary.csv")
items = []
for g in gens:
    k = clip_key(g["clip_id"])
    if args.source == "regen" and k in already:
        continue
    items.append({"key": k, "text": prompts[g["prompt_id"]]["text"], "wav": g["wav"],
                  "strata": (prompts[g["prompt_id"]]["category"], g["tts_model"])})

dest = OUT / "labeler"
(dest / "audio").mkdir(parents=True, exist_ok=True)
for it in items:
    shutil.copyfile(OUT / it["wav"], dest / "audio" / f"{it['key']}.wav")

rng = random.Random(1234)
rng.shuffle(items)


def page(name: str, subset: list[dict]):
    data = [{"key": i["key"], "text": i["text"]} for i in subset]
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__NAME__", name)
    (dest / f"{name}.html").write_text(html)
    print(f"{dest / f'{name}.html'}: {len(subset)} clips")


# Second-listener subset, stratified by (category, model).
second = []
if args.source == "original":
    by: dict = {}
    for it in items:
        by.setdefault(it["strata"], []).append(it)
    for group in by.values():
        n = max(1, round(args.second_fraction * len(group)))
        second.extend(rng.sample(group, n))
    rng.shuffle(second)

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label: __NAME__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--err:#b91c1c;--ok:#15803d;--acc:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--err:#f87171;--ok:#4ade80;--acc:#93c5fd}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 80px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:16px}
.prompt{font-size:20px;line-height:1.6;margin:8px 0 16px}
audio{width:100%}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on.ok{border-color:var(--ok);color:var(--ok);font-weight:600}
button.on.err{border-color:var(--err);color:var(--err);font-weight:600}
label{display:inline-flex;gap:6px;align-items:center;margin-right:14px}
input[type=text]{width:100%;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);box-sizing:border-box}
.mut{color:var(--mut);font-size:14px}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:13px}
</style></head><body><main>
<h1 style="margin:0">Listening labels: __NAME__</h1>
<p class="mut">Mark a clip <b>Error</b> if any prompt word is missing, repeated, replaced or cut off, or an extra word is spoken.
Odd pronunciation of a word that is still recognisable is <b>not</b> an error; note it instead.
Keys: <kbd>space</kbd> play/pause · <kbd>1</kbd> clean · <kbd>2</kbd> error · <kbd>o r s t i</kbd> toggle type · <kbd>→</kbd>/<kbd>←</kbd> next/prev</p>
<div class="mut" id="progress"></div>
<div class="card">
  <div class="mut" id="pos"></div>
  <div class="prompt" id="text"></div>
  <audio id="audio" controls></audio>
  <div class="row"><button id="b_ok" class="ok">1 · Clean</button><button id="b_err" class="err">2 · Error</button></div>
  <div class="row" id="types"></div>
  <div class="row"><input type="text" id="notes" placeholder="Notes (e.g. 'skipped “old wooden”', 'odd stress on Okonkwo')"></div>
  <div class="row"><button id="prev">← Prev</button><button id="next">Next →</button><span style="flex:1"></span><button id="export">Export CSV</button></div>
</div></main>
<script>
const DATA=__DATA__, NAME="__NAME__", TYPES=[["omission","o"],["repetition","r"],["substitution","s"],["truncation","t"],["insertion","i"]];
const store="labels_"+NAME; let L={}; try{L=JSON.parse(localStorage.getItem(store)||"{}")}catch(e){}
function save(){try{localStorage.setItem(store,JSON.stringify(L))}catch(e){}}
let i=0; const $=id=>document.getElementById(id);
$("types").innerHTML=TYPES.map(([t,k])=>`<label><input type="checkbox" data-t="${t}"> ${t} (${k})</label>`).join("");
function cur(){const k=DATA[i].key; return L[k]||(L[k]={error:"",types:[],notes:""})}
function render(){const d=DATA[i], l=cur();
  $("pos").textContent=`Clip ${i+1} of ${DATA.length} · ${d.key}`; $("text").textContent=d.text;
  $("audio").src="audio/"+d.key+".wav";
  $("b_ok").classList.toggle("on",l.error===0); $("b_err").classList.toggle("on",l.error===1);
  document.querySelectorAll("[data-t]").forEach(c=>c.checked=l.types.includes(c.dataset.t));
  $("notes").value=l.notes; const n=DATA.filter(x=>L[x.key]&&L[x.key].error!=="").length;
  $("progress").textContent=`${n} / ${DATA.length} labelled`;}
function set(e){cur().error=e; if(e===0)cur().types=[]; save(); render()}
function go(d){i=Math.max(0,Math.min(DATA.length-1,i+d)); render(); $("audio").play().catch(()=>{})}
$("b_ok").onclick=()=>set(0); $("b_err").onclick=()=>set(1);
$("prev").onclick=()=>go(-1); $("next").onclick=()=>go(1);
$("types").onchange=e=>{const t=e.target.dataset.t,l=cur(); l.types=e.target.checked?[...new Set([...l.types,t])]:l.types.filter(x=>x!==t); if(l.types.length)l.error=1; save(); render()};
$("notes").oninput=e=>{cur().notes=e.target.value; save()};
document.addEventListener("keydown",e=>{if(e.target.id==="notes")return;
  if(e.key===" "){e.preventDefault(); const a=$("audio"); a.paused?a.play():a.pause()}
  else if(e.key==="1")set(0); else if(e.key==="2")set(1);
  else if(e.key==="ArrowRight")go(1); else if(e.key==="ArrowLeft")go(-1);
  else{const m=TYPES.find(([t,k])=>k===e.key); if(m){const l=cur(); l.types=l.types.includes(m[0])?l.types.filter(x=>x!==m[0]):[...l.types,m[0]]; if(l.types.length)l.error=1; save(); render()}}});
$("export").onclick=()=>{const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const rows=["clip_key,error,types,notes",...DATA.map(d=>{const l=L[d.key]||{error:"",types:[],notes:""};return [d.key,l.error,l.types.join(";"),q(l.notes)].join(",")})];
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"})); a.download=NAME+".csv"; a.click()};
render();
</script></body></html>"""

if args.source == "original":
    page("primary", items)
    page("second", second)
else:
    page("regen", items)
print("Save exported CSVs into labels/ (primary.csv, second.csv; append regen.csv rows to primary.csv).")
