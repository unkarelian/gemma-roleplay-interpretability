"""Stage 2: render the static site from build/data/*.json and build/content/*.

    python3 build/render.py

Writes index.html, findings.html, atlas.html, viz.html and film.html next to this folder. Every link is
relative, so the site works from any sub-path (for example a GitHub Pages project site).
"""
import html, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
CONTENT = os.path.join(HERE, "content")
load = lambda n: json.load(open(os.path.join(DATA, n)))
read = lambda n: open(os.path.join(CONTENT, n), encoding="utf-8").read()

ATLAS = load("atlas.json")
ATLAS_KEYS = {f["key"] for f in ATLAS["features"]}
NP = "https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/"
TITLE = "Gemma role-play interpretability"
NAV = [("index.html", "Overview"), ("findings.html", "Findings"), ("atlas.html", "Atlas"),
       ("viz.html", "Layer views"), ("film.html", "Concept film")]


def macros(s):
    s = re.sub(r"\{\{np:(\d+)\}\}", lambda m: f'<a class="fref" href="{NP}{m.group(1)}" target="_blank" rel="noopener" '
               f'title="Layer-17 SAE latent {m.group(1)} on Neuronpedia">{m.group(1)}</a>', s)

    def tc(m):
        k = m.group(1)
        if k in ATLAS_KEYS:
            return f'<a class="fref mono" href="atlas.html#{k}" title="Open in the atlas">{k}</a>'
        return f'<span class="mono">{k}</span>'
    return re.sub(r"\{\{tc:(L\d+\.\d+)\}\}", tc, s)


def shell(fname, title, body, head="", scripts="", body_class="", main_class="wrap"):
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == fname else ""}>{t}</a>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="Interpretability research on Gemma 3 4B-it role-play: persona, refusals and users in danger, with Gemma Scope 2 transcoders.">
<meta name="color-scheme" content="light dark">
<link rel="stylesheet" href="assets/site.css">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
{head}
</head><body class="{body_class}">
<a class="skip" href="#main">Skip to content</a>
<header class="site-head"><div class="wrap">
<a class="brand" href="index.html">{TITLE}<small>Gemma 3 4B-it · Gemma Scope 2</small></a>
<nav class="site-nav" aria-label="Site">{nav}</nav>
</div></header>
<main id="main" class="{main_class}">
{body}
</main>
<footer class="site-foot"><div class="wrap">
<span>Interpretability research on Gemma 3 4B-it role-play. Feature labels are best guesses; see the caveats on each page.</span>
<span><a href="findings.html#caveats">Caveats</a> · <a href="atlas.html">Atlas</a> · <a href="findings.html">Findings</a></span>
</div></footer>
{scripts}
</body></html>
"""


def write(name, s):
    open(os.path.join(SITE, name), "w", encoding="utf-8").write(s)
    print("wrote", name, f"{len(s.encode()) / 1024:.0f} KB")


def js_data(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


# ---------------------------------------------------------------- index ---
INDEX_CSS = """<style>
.hero{padding:56px 0 28px;max-width:900px}
.hero h1{font-size:clamp(32px,4.4vw,54px);margin:.15em 0 .35em}
.hero-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}
.film-block{margin:8px 0 8px}
.findings-grid .fcard{display:flex;flex-direction:column;gap:2px;color:var(--text);transition:border-color .15s}
.findings-grid .fcard:hover{text-decoration:none;border-color:var(--accent)}
.fcard h3{font-size:16.5px;margin:.5em 0 .1em}.fcard p{font-size:14.5px;color:var(--text-2);margin:.4em 0 .8em;flex:1}
.fcard .stat{color:var(--accent)}.fcard .chip{align-self:flex-start}
.xcard{color:var(--text)}.xcard:hover{text-decoration:none;border-color:var(--accent)}
.xcard h3{margin:0 0 .3em;font-size:17px;color:var(--accent)}.xcard p{margin:0;font-size:14.5px;color:var(--text-2)}
</style>"""
write("index.html", shell("index.html", f"{TITLE}: does Gemma know it is role-playing?", read("index.html"), head=INDEX_CSS))

# ------------------------------------------------------------- findings ---
FIND_CSS = """<style>
.doc{display:grid;grid-template-columns:230px minmax(0,1fr);gap:48px;align-items:start}
.toc{position:sticky;top:76px;font-size:14px;max-height:calc(100vh - 96px);overflow:auto;padding-right:6px}
.toc ol{list-style:none;padding:0;margin:8px 0 0;counter-reset:t}
.toc li{margin:0}.toc a{display:block;padding:5px 10px;border-left:2px solid var(--border);color:var(--text-2)}
.toc a:hover{color:var(--text);text-decoration:none;border-color:var(--axis)}
.toc a.on{color:var(--accent);border-color:var(--accent);font-weight:600}
.doc .prose{max-width:860px}
.doc .prose>section>p,.doc .prose>section>ul,.doc .prose>section>h3,.doc .prose>section>dl,.doc .prose>section>.note{max-width:var(--prose)}
.doc section:first-child h2{margin-top:.4em}
.fref{font-variant-numeric:tabular-nums;font-weight:550}
.feat-table td:first-child{white-space:nowrap}
dl.defs dt{font-weight:650;margin-top:1.1em}dl.defs dd{margin:.25em 0 0;color:var(--text-2)}
@media (max-width:1000px){.doc{grid-template-columns:minmax(0,1fr);gap:0}.toc{position:static;max-height:none;border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin:8px 0 8px;background:var(--surface)}
 .toc ol{columns:2;column-gap:16px}.toc a{border-left:0;padding:3px 0}}
@media (max-width:560px){.toc ol{columns:1}}
</style>"""
FIND_JS = """<script>
(()=>{const links=[...document.querySelectorAll('.toc a')],secs=links.map(a=>document.querySelector(a.getAttribute('href')));
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')=='#'+e.target.id))}})},{rootMargin:'-20% 0px -70% 0px'});
secs.forEach(s=>s&&io.observe(s))})();
</script>"""
write("findings.html", shell("findings.html", f"Findings · {TITLE}", macros(read("findings.html")), head=FIND_CSS, scripts=FIND_JS))

# ---------------------------------------------------------------- atlas ---
atlas_data = dict(ATLAS, findings=json.load(open(os.path.join(CONTENT, "atlas_findings.json"))))
ATLAS_CSS = """<style>
.atlas-head{padding:32px 0 6px;display:flex;gap:24px;align-items:flex-end;justify-content:space-between;flex-wrap:wrap}
.atlas-head h1{margin:.1em 0 .2em}.atlas-head .lede{margin:0;font-size:17px}
.abar{position:sticky;top:57px;z-index:20;background:var(--bg);display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 0;border-bottom:1px solid var(--border);margin-bottom:18px}
.tabs{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}
.tabs button{flex:none;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;padding:6px 14px;font:inherit;font-size:14.5px;cursor:pointer}
.tabs button[aria-selected=true]{background:var(--text);color:var(--bg);border-color:var(--text)}
.abar input[type=search]{margin-left:auto;width:min(360px,100%);padding:8px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface);color:var(--text);font:inherit;font-size:15px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 20px}.stats .card{padding:12px 16px;min-width:150px}.stats .stat{font-size:24px}
.ov{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:32px;align-items:start}
.ov .side .card{margin-bottom:14px}.ov .side h3{margin:0 0 .4em;font-size:15.5px}.ov .side p,.ov .side li{font-size:14px;color:var(--text-2)}
.ov .side ul{padding-left:1.1em;margin:.3em 0}
.fgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.fgrid .card h3{margin:0 0 .35em;font-size:16px}.fgrid .big{font-size:15.5px;font-weight:650;color:var(--accent);margin:.2em 0 .4em}
.fgrid p{font-size:14.5px;color:var(--text-2);margin:.3em 0 .6em}
.cgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;align-items:start}
.cg h3{margin:0;font-size:16px}.cg .meta{font-size:13px;color:var(--muted);margin:2px 0 8px}
.strip{display:grid;grid-template-columns:repeat(34,1fr);gap:1px;margin:6px 0 10px}.strip i{height:12px;border-radius:2px;background:var(--plane)}
.frow{display:flex;gap:8px;align-items:baseline;padding:5px 6px;border-radius:6px;cursor:pointer;font-size:14px;width:100%;text-align:left;border:0;background:none;color:var(--text);font-family:inherit}
.frow:hover{background:var(--plane)}.frow .k{font-family:var(--mono);font-size:12.5px;color:var(--muted);min-width:76px}
.frow .l{flex:1}.conf{font-size:11.5px;color:var(--muted);white-space:nowrap}.conf.high{color:var(--green)}.conf.low{color:var(--amber)}
.more{font-size:13.5px;margin-top:4px;background:none;border:0;color:var(--accent);cursor:pointer;padding:4px 6px;font-family:inherit}
.lmap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
.lmap table{border-collapse:separate;border-spacing:2px;font-size:12px;font-variant-numeric:tabular-nums;margin:8px}
.lmap th{font-weight:500;color:var(--muted);padding:0 2px}.lmap th.g{text-align:right;white-space:nowrap;padding-right:8px;color:var(--text-2);font-size:13px;cursor:pointer}
.lmap td{width:22px;min-width:22px;height:22px;text-align:center;border-radius:4px;cursor:pointer;color:var(--text);background:var(--plane)}
.lmap td.z{cursor:default;color:transparent}.lmap td.sel{outline:2px solid var(--text);outline-offset:-1px}
.lmap td.hot{color:#fff}
#cell{margin-top:18px}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px;font-size:14px}
.filters select{font:inherit;font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text)}
.fcards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.fc{cursor:pointer;padding:14px 16px;text-align:left;font:inherit;color:var(--text);width:100%;display:flex;flex-direction:column;justify-content:flex-start;align-items:stretch}
.fc:hover{border-color:var(--accent)}.fc h3{margin:0 0 4px;font-size:15px;line-height:1.35}.fc .k{font-family:var(--mono);font-size:12.5px;color:var(--muted)}
.fc .c{font-size:13px;color:var(--muted);margin-top:4px}
#scrim{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:40;display:none}
#drawer{position:fixed;top:0;right:0;bottom:0;width:min(720px,100vw);background:var(--bg);border-left:1px solid var(--border);z-index:41;overflow-y:auto;transform:translateX(100%);transition:transform .2s ease;box-shadow:-8px 0 30px rgba(0,0,0,.12)}
#drawer.open{transform:none}#drawer .in{padding:22px 26px 60px}
#drawer h2{margin:.2em 0 .3em;font-size:22px}#drawer h4{margin:1.6em 0 .5em;font-size:14px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.dclose{position:sticky;top:0;float:right;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;width:36px;height:36px;font-size:20px;cursor:pointer;line-height:1}
.ex{font-family:var(--mono);font-size:12.5px;background:var(--plane);border:1px solid var(--border);border-radius:8px;padding:7px 10px;margin:6px 0;white-space:pre-wrap;word-break:break-word;line-height:1.5}
.ex mark{background:var(--accent-soft);color:var(--accent);border-radius:3px;padding:0 1px;font-weight:700;outline:1px solid var(--accent)}
.ex .v{float:right;color:var(--muted);font-family:var(--sans);font-size:12px;margin-left:8px}
.ex.wh{font-family:var(--sans);color:var(--muted);font-style:italic}
.bars{display:grid;grid-template-columns:auto 1fr auto;gap:5px 10px;align-items:center;font-size:13px}
.bars .n{color:var(--text-2)}.bars .b{height:10px;border-radius:3px;background:var(--pole0);min-width:1px}.bars .val{font-variant-numeric:tabular-nums;color:var(--muted);text-align:right}
table.ct{font-size:13.5px}table.ct td{padding:6px 10px}
@media (max-width:1000px){.ov{grid-template-columns:1fr}.fgrid{grid-template-columns:1fr}}
@media (max-width:760px){.abar{top:0}.abar input[type=search]{margin-left:0;width:100%}.cgrid,.fcards{grid-template-columns:minmax(0,1fr)}#drawer .in{padding:16px 16px 60px}}
</style>"""
ATLAS_BODY = """
<div class="atlas-head"><div><p class="kicker">Persona atlas</p><h1>Persona atlas</h1>
<p class="lede">250 Gemma Scope 2 transcoder features that separate our matched conditions, labelled and grouped into concepts.</p></div></div>
<div class="abar"><div class="tabs" role="tablist" id="tabs"></div>
<input type="search" id="q" placeholder="Search labels, concepts or a key (L17.317)" aria-label="Search features"></div>
<div id="view"></div>
<div id="scrim"></div><aside id="drawer" aria-hidden="true" aria-label="Feature details"><div class="in" id="dbody"></div></aside>
"""
ATLAS_JS = r"""<script>
const D=__DATA__;
const TABS=[["overview","Overview"],["concepts","Concepts"],["layers","Layer map"],["features","Features"]];
const NL=34, CONF={high:0,medium:1,low:2};
let tab="overview", cellSel=null, open=null;
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tok=s=>s.replace(/<start_of_turn>/g,'‹turn›').replace(/<end_of_turn>/g,'‹end›').replace(/<bos>/g,'‹bos›');
const exh=s=>esc(tok(s)).replace(/\[\[(.*?)\]\]/gs,'<mark>$1</mark>');
const posName={last:"decision token",reply:"reply tokens",message:"user message",system:"system prompt"};
const F=D.features, byKey=Object.fromEntries(F.map(f=>[f.key,f]));
function chips(f){return f.evidence.map(e=>`<span class="chip ${e}">${e}</span>`).join('')+(f.dim443>0.5?`<span class="chip flag" title="Most of its decoder writes residual dimension 443">dim-443 writer</span>`:'')+`<span class=chip>${esc(f.kind)}</span><span class=chip>${f.conf} confidence</span>`}
function tabsUI(){$('#tabs').innerHTML=TABS.map(([k,t])=>`<button role=tab aria-selected=${k==tab} onclick="go('${k}')">${t}</button>`).join('')}
function go(k,keepHash){tab=k;tabsUI();render();if(!keepHash)history.replaceState(null,'','#'+k);window.scrollTo({top:0})}
function fcard(f){return `<button class="card fc" onclick="show('${f.key}')"><div class=k>${f.key}</div><h3>${esc(f.label)}</h3><div>${chips(f)}</div><div class=c>${esc(f.concept)}</div></button>`}
function frow(f){return `<button class=frow onclick="show('${f.key}')"><span class=k>${f.key}</span><span class=l>${esc(f.label)}</span><span class="conf ${f.conf}">${f.conf}</span></button>`}
function groupsPresent(){return D.groups.filter(g=>F.some(f=>f.group==g))}
const heat=(n,max)=>{if(!n)return'';const a=0.18+0.82*Math.min(1,n/max);return `background:color-mix(in srgb,var(--pole0) ${Math.round(a*100)}%,var(--plane))`};
function render(){const m=$('#view');
 if(tab=="overview"){
  const cc={high:0,medium:0,low:0};F.forEach(f=>cc[f.conf]++);
  m.innerHTML=`<div class=stats><div class=card><div class=stat>${F.length}</div><div class="muted small">labelled features</div></div>
   <div class=card><div class=stat>${groupsPresent().length}</div><div class="muted small">concept groups</div></div>
   <div class=card><div class=stat>0–33</div><div class="muted small">layers covered</div></div>
   <div class=card><div class=stat>${cc.high} · ${cc.medium} · ${cc.low}</div><div class="muted small">high · medium · low confidence</div></div></div>
  <div class=ov><div><h2 style="margin-top:0">Key findings</h2><div class=fgrid>${D.findings.map(f=>`<div class=card><h3>${esc(f.title)}</h3>${f.big?`<div class=big>${esc(f.big)}</div>`:''}<p>${esc(f.body)}</p>${(f.evidence||[]).map(e=>`<span class="chip ${e}">${e}</span>`).join('')} <a class=small href="${f.link}">In the write-up →</a></div>`).join('')}</div></div>
  <div class=side><div class=card><h3>How the atlas was built</h3><p>A corpus of 6,733 condition-tagged conversations was run through Gemma 3 4B-it, and every Gemma Scope 2 transcoder feature (16k per layer, 34 layers) was recorded at each position. Features were selected where matched conditions differ (for example character vs plain assistant, real vs story crisis, harmful vs benign twin): found on one half of the characters and prompts, and confirmed on the other half.</p>
   <p>Labels were written by Claude from three sources: each feature's top examples in our corpus, Gemma Scope's own example texts, and its output tokens. They are best guesses, each with a confidence level.</p></div>
   <div class=card><h3>Reading a card</h3><ul>
   <li><span class="chip associated">associated</span> differs between matched conditions, confirmed on held-out characters.</li>
   <li><span class="chip attributed">attributed</span> appears on a path in one of the attribution graphs.</li>
   <li><span class="chip intervened">intervened</span> changing it changes behaviour (findings only).</li>
   <li><span class="chip flag">dim-443 writer</span> most of its decoder writes residual dimension 443, a massive-activation dimension that later normalisation largely absorbs. The label describes what it reads.</li>
   <li><b>Effect size d</b> is the paired effect on the held-out confirmation half; "A vs B" are mean activations in the two arms.</li></ul></div>
   <div class=card><h3>What is not shown</h3><p>Snippets from conversations with a user in danger are replaced by a short description and the firing token. For harmful requests only the request is shown, never the model's reply. Links are removed.</p></div></div></div>`}
 if(tab=="concepts"){const by={};F.forEach(f=>(by[f.group]=by[f.group]||[]).push(f));
  m.innerHTML=`<p class="muted small" style="margin-top:0">Each group gathers features with a shared concept. The strip shows how many of its features sit at each layer (0 → 33). Click a feature for its card.</p><div class=cgrid>`+groupsPresent().map(g=>{const fs=by[g].sort((a,b)=>CONF[a.conf]-CONF[b.conf]||a.layer-b.layer);
   const lay=fs.map(f=>f.layer),cnt=Array(NL).fill(0);lay.forEach(l=>cnt[l]++);const mx=Math.max(...cnt);
   const id='g'+D.groups.indexOf(g);
   return `<div class="card cg"><h3>${esc(g)}</h3><div class=meta>${fs.length} features · layers ${Math.min(...lay)}–${Math.max(...lay)}</div>
   <div class=strip title="features per layer">${cnt.map((n,l)=>`<i style="${heat(n,mx)}" title="layer ${l}: ${n}"></i>`).join('')}</div>
   <div id=${id}>${fs.slice(0,8).map(frow).join('')}</div>${fs.length>8?`<button class=more onclick="expand(this,${D.groups.indexOf(g)})">Show all ${fs.length}</button>`:''}</div>`}).join('')+`</div>`}
 if(tab=="layers"){const G=groupsPresent();const cnt=(l,g)=>F.filter(f=>f.layer==l&&f.group==g).length;let mx=1;G.forEach(g=>{for(let l=0;l<NL;l++)mx=Math.max(mx,cnt(l,g))});
  let h=`<p class="muted small" style="margin-top:0">Number of mapped features per concept group (rows) and layer (columns). Click a cell for its features, or a group name for the whole group.</p><div class=lmap><table><tr><th></th>${[...Array(NL).keys()].map(l=>`<th>${l}</th>`).join('')}</tr>`;
  G.forEach((g,gi)=>{h+=`<tr><th class=g onclick="cell(null,${gi})">${esc(g)}</th>`;for(let l=0;l<NL;l++){const n=cnt(l,g);const sel=cellSel&&cellSel[0]==l&&cellSel[1]==gi;
   h+=n?`<td data-c="${l}-${gi}" class="${n/mx>0.45?'hot':''}${sel?' sel':''}" style="${heat(n,mx)}" onclick="cell(${l},${gi})" title="${esc(g)} · layer ${l}: ${n}">${n}</td>`:`<td class=z></td>`}h+='</tr>'});
  m.innerHTML=h+`</table></div><div id=cell></div>`;if(cellSel)cell(...cellSel,true)}
 if(tab=="features"){const kinds=[...new Set(F.map(f=>f.kind))].sort();
  m.innerHTML=`<div class=filters><select id=fg aria-label="Concept group"><option value="">All groups</option>${groupsPresent().map(g=>`<option>${esc(g)}</option>`).join('')}</select>
   <select id=fk aria-label="Kind"><option value="">All kinds</option>${kinds.map(k=>`<option>${esc(k)}</option>`).join('')}</select>
   <select id=fcf aria-label="Confidence"><option value="">Any confidence</option><option>high</option><option>medium</option><option>low</option></select>
   <select id=fs aria-label="Sort"><option value=layer>Sort by layer</option><option value=conf>Sort by confidence</option></select><span class="muted" id=fn></span></div><div class=fcards id=fl></div>`;
  ['fg','fk','fcf','fs'].forEach(i=>$('#'+i).onchange=filt);filt()}}
function expand(b,gi){b.previousElementSibling.innerHTML=groupList(D.groups[gi]);b.remove()}
function groupList(g){return F.filter(f=>f.group==g).sort((a,b)=>CONF[a.conf]-CONF[b.conf]||a.layer-b.layer).map(frow).join('')}
function cell(l,gi,quiet){const g=groupsPresent()[gi];cellSel=[l,gi];const fs=F.filter(f=>f.group==g&&(l===null||f.layer==l)).sort((a,b)=>a.layer-b.layer);
 if(!quiet){document.querySelectorAll('.lmap td.sel').forEach(t=>t.classList.remove('sel'));document.querySelector(`.lmap td[data-c="${l}-${gi}"]`)?.classList.add('sel')}
 $('#cell').innerHTML=`<h3>${esc(g)}${l!==null?` · layer ${l}`:''}</h3><div class=fcards>${fs.map(fcard).join('')}</div>`;if(!quiet)$('#cell').scrollIntoView({behavior:'smooth',block:'nearest'})}
function filt(){const q=($('#q').value||'').toLowerCase().trim(),g=$('#fg')?.value,k=$('#fk')?.value,c=$('#fcf')?.value,s=$('#fs')?.value||'layer';
 let fs=F.filter(f=>(!q||(f.key+' '+f.label+' '+f.concept+' '+f.group+' '+f.kind+' '+f.note).toLowerCase().includes(q))&&(!g||f.group==g)&&(!k||f.kind==k)&&(!c||f.conf==c));
 fs=fs.sort(s=='conf'?(a,b)=>CONF[a.conf]-CONF[b.conf]||a.layer-b.layer:(a,b)=>a.layer-b.layer||a.feat-b.feat);
 $('#fn').textContent=`${fs.length} of ${F.length}`;$('#fl').innerHTML=fs.map(fcard).join('')||'<p class=muted>No features match.</p>'}
function exBox(e,withV){if(e.t===null||e.t===undefined)return `<div class="ex wh">${withV&&e.v!=null?`<span class=v>${e.v}</span>`:''}Text withheld: ${esc(e.why)}. Fires on “${esc(tok(e.tok))}” ${esc(e.pos)}.</div>`;
 return `<div class=ex>${withV&&e.v!=null?`<span class=v>${e.v}</span>`:''}${exh(e.t)}</div>`}
function show(k,noHash){const f=byKey[k];if(!f)return;open=k;
 const bl=Object.entries(f.blocks||{}).filter(([b,v])=>D.blocks[b]).sort((a,b)=>b[1]-a[1]);const bmax=Math.max(1e-9,...bl.map(x=>Math.abs(x[1])));
 let h=`<button class=dclose onclick="hide()" aria-label="Close">×</button><div class="mono muted">${f.key} · layer ${f.layer}, feature ${f.feat}</div><h2>${esc(f.label)}</h2><div>${chips(f)}</div>
 <p class=small><b>Concept:</b> ${esc(f.concept)} · <b>Group:</b> ${esc(f.group)}${f.note?`<br><b>Note:</b> ${esc(f.note)}`:''}</p>
 <h4>Where it differs</h4><div class=table-wrap><table class="data ct"><tr><th>Contrast (arm A vs arm B)</th><th>Position</th><th class=num>d</th><th class=num>A vs B</th></tr>${f.contrasts.map(c=>`<tr><td>${esc(c.c)}</td><td>${posName[c.role]||c.role}</td><td class=num>${c.d}</td><td class=num>${c.a} vs ${c.b}</td></tr>`).join('')}</table></div>
 <h4>Top examples in our corpus</h4>${f.top.map(e=>exBox(e,true)).join('')}`;
 if(f.neg.length)h+=`<h4>A matched example where it is silent</h4>${f.neg.map(e=>exBox(e)).join('')}`;
 if(f.gs.length)h+=`<h4>Gemma Scope's own top examples</h4>${f.gs.map(s=>`<div class=ex>${exh(s)}</div>`).join('')}<p class="small muted">Output tokens it pushes up: ${esc(f.logits.join(' · '))}</p>`;
 if(bl.length)h+=`<h4>Mean activation by corpus set</h4><div class=bars>${bl.map(([b,v])=>`<span class=n>${esc(D.blocks[b])}</span><span><span class=b style="display:block;width:${Math.max(0.5,100*Math.abs(v)/bmax)}%"></span></span><span class=val>${v}</span>`).join('')}</div>`;
 if(f.np.length)h+=`<h4>Closest Neuronpedia feature (262k transcoder)</h4>${f.np.map(n=>`<div class=small><a href="${esc(n.u)}" target=_blank rel=noopener>${esc(n.l||'(no label)')}</a> <span class=muted>decoder cosine ${n.c}${n.c<0.5?' (probably not the same feature)':''}</span></div>`).join('')}`;
 $('#dbody').innerHTML=h;$('#drawer').classList.add('open');$('#drawer').setAttribute('aria-hidden','false');$('#scrim').style.display='block';$('#drawer').scrollTop=0;
 if(!noHash)history.replaceState(null,'','#'+k);$('.dclose').focus({preventScroll:true})}
function hide(){open=null;$('#drawer').classList.remove('open');$('#drawer').setAttribute('aria-hidden','true');$('#scrim').style.display='none';history.replaceState(null,'','#'+tab)}
$('#scrim').onclick=hide;addEventListener('keydown',e=>{if(e.key=='Escape'&&open)hide()});
$('#q').addEventListener('input',()=>{if(tab!='features')go('features');filt()});
function route(){const h=decodeURIComponent(location.hash.slice(1));if(byKey[h]){if(tab!='features'&&!document.querySelector('#view').innerHTML)go('features',true);show(h,true)}else if(TABS.some(t=>t[0]==h))go(h,true);}
addEventListener('hashchange',route);
tabsUI();render();route();
</script>"""
write("atlas.html", shell("atlas.html", f"Persona atlas · {TITLE}", ATLAS_BODY, head=ATLAS_CSS,
                          scripts=ATLAS_JS.replace("__DATA__", js_data(atlas_data))))

# ------------------------------------------------------------ layer views ---
V = load("viz.json")
G = V["groups"]
INFO = {
    "refusal": dict(label="Refusal", pole0="like the harmless twin", pole1="like a harmful request",
                    what=f"How strongly each token, at each layer, looks like a harmful request rather than a benign one. The direction is fit on {G['refusal'][0]} harmful and {G['refusal'][1]} benign XSTest prompts given to the plain assistant."),
    "persona": dict(label="Persona", pole0="like the plain assistant", pole1="like a character",
                    what=f"How strongly each token looks like the model playing a character rather than being the plain assistant, with tone removed. The direction is fit on {G['persona'][0]} character and {G['persona'][1]} plain-assistant prompts from the tone-swap set (the same identities told to be kind, hostile or neutral)."),
    "user harm": dict(label="User in danger", pole0="like an ordinary message", pole1="like a user in danger",
                      what=f"How strongly each token looks like a real user in danger (medical emergency, pills, distress) rather than an ordinary message. The direction is fit on {G['user harm'][0]} crisis and {G['user harm'][1]} harmless messages to the plain assistant, a small group."),
}
for k, sc in V["scenes"].items():
    cv = [row[-1] for row in sc["case"]["values"][k]]; nv = [row[-1] for row in sc["control"]["values"][k]]
    n = len(cv)
    gap = next((L for L in range(n) if all(cv[j] - nv[j] >= 0.5 for j in range(L, min(n, L + 4)))), None)

    def settle(lens):
        w = lens[-1][0][0]
        L = next(L for L in range(n) if all(lens[j][0][0] == w for j in range(L, n)))
        return w.strip() or repr(w), L
    wc, lc = settle(sc["case"]["lens"]); wn, ln = settle(sc["control"]["lens"])
    INFO[k]["caption"] = ((f"The two prompts look alike at the last token until about layer {gap}; from there this one reads as "
                           f"{INFO[k]['pole1'].replace('like ', '', 1)} while the control falls back. " if gap is not None else
                           "The two prompts never fully separate at the last token. ")
                          + f"The first word of the reply settles at layer {lc}: “{wc}”, where the control settles on “{wn}” (layer {ln}).")
VIZ_CSS = """<style>
.viz-root{--surface-1:var(--surface);--text-primary:var(--text);--text-secondary:var(--text-2);--s-case:var(--pole1);--s-ctrl:var(--pole0)}
.vtabs{display:flex;gap:6px;overflow-x:auto;margin:6px 0 4px}.vtabs button{flex:none;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;padding:7px 16px;font:inherit;font-size:14.5px;cursor:pointer}
.vtabs button[aria-selected=true]{background:var(--text);color:var(--bg);border-color:var(--text)}
.vcard{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:14px 0}
.lead{font-size:16.5px;margin:0 0 6px}.sub{color:var(--text-2);font-size:13.5px;margin:0}
.vgrid{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(0,1fr);gap:14px}.vgrid>.vcard,.vgrid>div>.vcard:first-child{margin-top:0}
@media (max-width:900px){.vgrid{grid-template-columns:minmax(0,1fr)}}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:999px;overflow:hidden}.seg button{border:0;background:none;color:var(--text);font:inherit;font-size:13.5px;padding:6px 14px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--text);color:var(--bg)}
.play{border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;padding:6px 14px;font:inherit;font-size:13.5px;cursor:pointer}
input[type=range]{flex:1;min-width:140px;accent-color:var(--text)}
.layer{font-variant-numeric:tabular-nums;font-weight:600;min-width:74px}
.heat{position:relative;overflow-x:auto}.heat canvas{display:block}
.legend{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--text-2);margin-top:8px;flex-wrap:wrap}
.ramp{width:160px;height:10px;border-radius:5px}
.key{display:inline-flex;align-items:center;gap:6px}.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.lens{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;font-size:14px;align-items:baseline}
.lens .w{font-family:var(--mono);font-size:14.5px;background:var(--plane);border-radius:6px;padding:1px 6px}
.lens .p{color:var(--text-2);font-size:12px}
.msg{font-size:13.5px;color:var(--text-2);white-space:pre-wrap;margin:8px 0 0}
.reply{font-size:14px;white-space:pre-wrap;background:var(--plane);border-radius:10px;padding:10px 12px;margin:6px 0 0}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12px;display:none;z-index:50;box-shadow:0 4px 14px rgba(0,0,0,.12)}
.k{color:var(--text-2)}details{margin-top:10px}summary{cursor:pointer;color:var(--text-2);font-size:13.5px}
table.vt{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:6px}table.vt td,table.vt th{padding:2px 10px;text-align:right;border-bottom:1px solid var(--grid)}
</style>"""
VIZ_BODY = """
<div class="page-head"><p class="kicker">Layer views</p><h1>A prompt going through Gemma's 34 layers</h1>
<p class="lede">Three showcase prompts, each beside a matched control. For every token and every layer, the heatmap shows where the residual stream sits on one direction: refusal, persona, or a user in danger.</p></div>
<div class="vtabs" role="tablist" id="tabs"></div>
<div id="vmain"></div><div id="tip"></div>
<p class="small muted prose">How the directions are made: for each concept, a mean difference at the last prompt token between two matched groups of corpus prompts (the discovery half, or all of the corpus where a group was too small), with residual dimension 443 zeroed. Each layer is calibrated separately so 0 = the control group's mean and 1 = the concept group's mean. The showcase prompts are scored on these directions; the logit lens reads the residual at each layer through the final norm and unembedding. Replies are greedy, 60 tokens.</p>
"""
VIZ_JS = r"""<script>
const D=__DATA__, INFO=__INFO__;
const NL=D.n_layers, order=["refusal","persona","user harm"], TABN={"refusal":"Refusal · a harmful request","persona":"Persona · becoming a character","user harm":"A user in danger"};
let cur=order[0], side="case", layer=NL-1, timer=null;
const $=s=>document.querySelector(s);
const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
function hex2rgb(h){h=h.replace('#','');return[0,2,4].map(i=>parseInt(h.slice(i,i+2),16)/255)}
const lin=c=>c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4), gam=c=>c<=.0031308?12.92*c:1.055*Math.pow(c,1/2.4)-.055;
function toLab([r,g,b]){r=lin(r);g=lin(g);b=lin(b);const l=Math.cbrt(.4122214708*r+.5363325363*g+.0514459929*b),m=Math.cbrt(.2119034982*r+.6806995451*g+.1073969566*b),s=Math.cbrt(.0883024619*r+.2817188376*g+.6299787005*b);
 return[.2104542553*l+.793617785*m-.0040720468*s,1.9779984951*l-2.428592205*m+.4505937099*s,.0259040371*l+.7827717662*m-.808675766*s]}
function fromLab([L,a,b]){const l=(L+.3963377774*a+.2158037573*b)**3,m=(L-.1055613458*a-.0638541728*b)**3,s=(L-.0894841775*a-1.291485548*b)**3;
 return[4.0767416621*l-3.3077115913*m+.2309699292*s,-1.2684380046*l+2.6097574011*m-.3413193965*s,-.0041960863*l-.7034186147*m+1.707614701*s].map(c=>Math.round(255*Math.min(1,Math.max(0,gam(c)))))}
let P0,P1,PM;function pal(){P0=toLab(hex2rgb(css('--pole0')));P1=toLab(hex2rgb(css('--pole1')));PM=toLab(hex2rgb(css('--mid')))}
function col(v){const t=Math.max(0,Math.min(1,v));let a,b,u;if(t<.5){a=P0;b=PM;u=t/.5}else{a=PM;b=P1;u=(t-.5)/.5}
 const c=fromLab(a.map((x,i)=>x+(b[i]-x)*u));return`rgb(${c})`}
function tabs(){$('#tabs').innerHTML=order.map(k=>`<button role=tab aria-selected=${k==cur} onclick="go('${k}')">${TABN[k]}</button>`).join('')}
function go(k){cur=k;side="case";layer=NL-1;stop();render();tabs();history.replaceState(null,'','#'+k.replace(' ','-'))}
const showTok=t=>t.replace(/\n/g,'⏎').replace(/<start_of_turn>/,'‹turn›').replace(/<end_of_turn>/,'‹end›').replace(/<bos>/,'‹bos›');
function render(){
 const S=D.scenes[cur],I=INFO[cur];
 $('#vmain').innerHTML=`
 <div class=vcard><p class=lead><b>${S.title}.</b> ${I.caption}</p><p class=sub>${I.what} 0 = ${I.pole0}, 1 = ${I.pole1}, calibrated per layer.</p></div>
 <div class=vcard><div class=row>
   <div class=seg role=group aria-label="Prompt">${['case','control'].map(s=>`<button aria-pressed=${s==side} onclick="side='${s}';render()">${s=='case'?S.title:'Matched control'}</button>`).join('')}</div>
   <button class=play id=play onclick="toggle()">▶ Play</button>
   <input type=range id=lay min=0 max=${NL-1} value=${layer} aria-label=Layer oninput="layer=+this.value;update()"><span class=layer id=laylab></span></div>
   <div class=msg id=msgs></div></div>
 <div class=vgrid>
   <div class=vcard><b>Every token, every layer</b><p class=sub>Rows are layers (bottom = first layer, top = last); columns are the prompt's tokens. Press play to watch the prompt rise through the model; hover for values, click to pick a layer.</p>
     <div class=heat><canvas id=hm></canvas></div>
     <div class=legend><span>${I.pole0}</span><span class=ramp id=ramp></span><span>${I.pole1}</span></div></div>
   <div>
     <div class=vcard><b>The last token, layer by layer</b><p class=sub>The model decides how to answer at the last prompt token. Red: ${S.title.toLowerCase()}; blue: the matched control.</p><svg id=line role=img aria-label="value at the last token by layer"></svg>
       <div class=legend><span class=key><span class=sw style="background:var(--s-case)"></span>${S.title}</span><span class=key><span class=sw style="background:var(--s-ctrl)"></span>Matched control</span></div></div>
     <div class=vcard><b>What it is about to say</b> <span class=sub>(logit lens at this layer)</span><div class=lens id=lens style="margin-top:8px"></div></div>
   </div></div>
 <div class=vcard><b>What it actually said</b> <span class=sub>(first 60 tokens)</span><div class=vgrid style="margin-top:8px;grid-template-columns:1fr 1fr"><div><span class=k>${S.title}</span><div class=reply>${esc(S.case.reply)}</div></div><div><span class=k>Matched control</span><div class=reply>${esc(S.control.reply)}</div></div></div>
   <details><summary>Table view: last-token values by layer</summary><table class=vt><tr><th>Layer</th><th>${S.title}</th><th>Control</th><th>Gap</th></tr>${S.case.values[cur].map((r,L)=>`<tr><td>${L}</td><td>${r[r.length-1].toFixed(2)}</td><td>${S.control.values[cur][L][S.control.values[cur][L].length-1].toFixed(2)}</td><td>${(r[r.length-1]-S.control.values[cur][L][S.control.values[cur][L].length-1]).toFixed(2)}</td></tr>`).join('')}</table></details></div>`;
 pal(); $('#ramp').style.background=`linear-gradient(90deg,${[0,.25,.5,.75,1].map(col).join(',')})`;
 const M=(side=='case'?S.case_msgs:S.control_msgs)||[];$('#msgs').textContent=M.map(m=>`${m.role}: ${m.content}`).join('\n');
 if(innerWidth<900)$('.vcard:last-child .vgrid').style.gridTemplateColumns='minmax(0,1fr)';
 update();
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function update(){
 const S=D.scenes[cur],X=S[side],V=X.values[cur],T=X.tokens.length;
 $('#lay').value=layer;$('#laylab').textContent=`Layer ${layer}`;
 const cv=$('#hm'),cw=Math.max(16,Math.min(40,Math.floor((cv.parentElement.clientWidth-40)/T))),ch=10,lw=34,th=110,W=lw+cw*T,H=ch*NL+th;
 const dpr=devicePixelRatio||1;cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);
 for(let L=0;L<NL;L++)for(let t=0;t<T;t++){const y=(NL-1-L)*ch;g.globalAlpha=L<=layer?1:.12;g.fillStyle=col(V[L][t]);g.fillRect(lw+t*cw+1,y+1,cw-2,ch-2)}
 g.globalAlpha=1;g.strokeStyle=css('--text');g.lineWidth=1.5;g.strokeRect(lw+.5,(NL-1-layer)*ch+.5,cw*T-1,ch-1);
 g.fillStyle=css('--muted');g.font='11px system-ui';g.textAlign='right';g.textBaseline='middle';
 for(const L of [0,8,16,24,33])g.fillText(L,lw-6,(NL-1-L)*ch+ch/2);
 g.save();g.textAlign='right';g.fillStyle=css('--text-2');g.font='12px ui-monospace,Menlo,monospace';
 for(let t=0;t<T;t++){g.save();g.translate(lw+t*cw+cw/2,NL*ch+6);g.rotate(-Math.PI/3);g.fillStyle=X.roles[t]=='template'?css('--muted'):css('--text');g.fillText(showTok(X.tokens[t]).slice(0,14),0,0);g.restore()}
 g.restore();
 cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),t=Math.floor((e.clientX-r.left-lw)/cw),L=NL-1-Math.floor((e.clientY-r.top)/ch);
   if(t<0||t>=T||L<0||L>=NL){hide();return}tip(`<b>${esc(showTok(X.tokens[t]))}</b> · layer ${L}<br>${INFO[cur].label}: ${V[L][t].toFixed(2)}`,e.clientX,e.clientY)};
 cv.onmouseleave=hide; cv.onclick=e=>{const r=cv.getBoundingClientRect(),L=NL-1-Math.floor((e.clientY-r.top)/ch);if(L>=0&&L<NL){layer=L;stop();update()}};
 line(); lens();
}
function line(){
 const S=D.scenes[cur],a=S.case.values[cur].map(r=>r[r.length-1]),b=S.control.values[cur].map(r=>r[r.length-1]);
 const svg=$('#line'),W=svg.parentElement.clientWidth-36,H=190,l=30,r=8,t=8,bt=22;svg.setAttribute('width',W);svg.setAttribute('height',H);
 const lo=Math.min(-.2,...a,...b),hi=Math.max(1.2,...a,...b),x=L=>l+(W-l-r)*L/(NL-1),y=v=>t+(H-t-bt)*(1-(v-lo)/(hi-lo));
 const path=v=>v.map((q,L)=>`${L?'L':'M'}${x(L).toFixed(1)},${y(q).toFixed(1)}`).join('');
 let s='';for(const g of [0,.5,1])s+=`<line x1=${l} x2=${W-r} y1=${y(g)} y2=${y(g)} stroke="var(--grid)" stroke-width=1 ${g==.5?'stroke-dasharray="3 3"':''} /><text x=${l-5} y=${y(g)+4} font-size=11 text-anchor=end fill="var(--muted)">${g}</text>`;
 for(const L of [0,8,16,24,33])s+=`<text x=${x(L)} y=${H-6} font-size=11 text-anchor=middle fill="var(--muted)">${L}</text>`;
 s+=`<line x1=${x(layer)} x2=${x(layer)} y1=${t} y2=${H-bt} stroke="var(--axis)" stroke-width=1 />`;
 s+=`<path d="${path(b)}" fill=none stroke="var(--s-ctrl)" stroke-width=2 stroke-linejoin=round /><path d="${path(a)}" fill=none stroke="var(--s-case)" stroke-width=2 stroke-linejoin=round />`;
 s+=`<circle cx=${x(layer)} cy=${y(b[layer])} r=4 fill="var(--s-ctrl)" stroke="var(--surface)" stroke-width=2 /><circle cx=${x(layer)} cy=${y(a[layer])} r=4 fill="var(--s-case)" stroke="var(--surface)" stroke-width=2 />`;
 s+=`<rect x=${l} y=${t} width=${W-l-r} height=${H-t-bt} fill=transparent id=hit />`;svg.innerHTML=s;
 svg.querySelector('#hit').onmousemove=e=>{const R=svg.getBoundingClientRect(),L=Math.round((e.clientX-R.left-l)/(W-l-r)*(NL-1));if(L<0||L>=NL)return;
   tip(`Layer ${L}<br><span style="color:var(--s-case)">●</span> ${a[L].toFixed(2)} · <span style="color:var(--s-ctrl)">●</span> ${b[L].toFixed(2)}`,e.clientX,e.clientY)};
 svg.querySelector('#hit').onmouseleave=hide;svg.querySelector('#hit').onclick=e=>{const R=svg.getBoundingClientRect();layer=Math.max(0,Math.min(NL-1,Math.round((e.clientX-R.left-l)/(W-l-r)*(NL-1))));stop();update()};
}
function lens(){const S=D.scenes[cur],f=x=>x.map(([w,p])=>`<span class=w>${esc(showTok(w))}</span> <span class=p>${Math.round(p*100)}%</span>`).join(' ');
 $('#lens').innerHTML=`<span class=k>${esc(S.title)}</span><span>${f(S.case.lens[layer])}</span><span class=k>Control</span><span>${f(S.control.lens[layer])}</span>`}
function tip(h,x,y){const t=$('#tip');t.innerHTML=h;t.style.display='block';t.style.left=Math.min(x+12,innerWidth-200)+'px';t.style.top=(y+12)+'px'}
function hide(){$('#tip').style.display='none'}
function toggle(){if(timer){stop();return}if(layer>=NL-1)layer=0;$('#play').textContent='❚❚ Pause';timer=setInterval(()=>{if(layer>=NL-1){stop();return}layer++;update()},180)}
function stop(){clearInterval(timer);timer=null;const p=$('#play');if(p)p.textContent='▶ Play'}
let rz;addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(render,120)});matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>render());
{const h=decodeURIComponent(location.hash.slice(1)).replace('-',' ');if(order.includes(h))cur=h}
tabs();render();
</script>"""
write("viz.html", shell("viz.html", f"Layer views · {TITLE}", VIZ_BODY, head=VIZ_CSS, body_class="viz-root",
                        scripts=VIZ_JS.replace("__DATA__", js_data(V)).replace("__INFO__", js_data(INFO))))

# ---------------------------------------------------------------- film ---
FILM = load("film.json")
FILM_CSS = """<style>
.fbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
.fbar button{border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;padding:7px 15px;font:inherit;font-size:14px;cursor:pointer}
.fbar button[aria-pressed=true]{background:var(--text);color:var(--bg);border-color:var(--text)}
.fbar input[type=range]{flex:1;min-width:160px;accent-color:var(--text)}
.fbar .t{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13.5px;min-width:88px;text-align:right}
canvas#cv{width:100%;height:auto;display:block;border-radius:12px;background:#121211;border:1px solid var(--border)}
.narrow-only{display:none}@media (max-width:760px){.narrow-only{display:block}}
.vids h3{font-size:16px;margin:0 0 8px}.vids figure{margin:0}
</style>"""
FILM_BODY = """
<div class="page-head"><p class="kicker">Concept film</p><h1>Concepts lighting up, layer by layer</h1>
<p class="lede">Watch a prompt pass through Gemma 3 4B-it. At each layer the panel shows up to three named transcoder features that are active, which prompt tokens they fire on, and the refusal, persona and user-danger directions at the last prompt token.</p></div>
<div class="fbar" id="scenes" role="group" aria-label="Scene"></div>
<div class="fbar"><span id="sides"></span><button id="play">▶ Play</button><input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Time"><span class="t" id="tt"></span></div>
<canvas id="cv" width="1280" height="720" aria-label="Animation of concepts activating layer by layer"></canvas>
<p class="small muted narrow-only">The animation is drawn at 1280×720. On a phone, turn it sideways or use the rendered films below.</p>
<p class="small muted prose">Each caption is a Gemma Scope 2 transcoder feature (16k per layer) active at that layer, named from its examples; low-confidence and unclear labels are left out, and features with meaning are preferred over format and word-identity features. Meters are the directions from the <a href="viz.html">layer views</a> at the last prompt token (0 = like the matched control, 1 = like the concept). The bottom line is the logit lens's best guess for the first word of the reply. Labels are best guesses from examples, not proofs.</p>

<section class="vids" aria-labelledby="vid-h">
<h2 id="vid-h">The rendered films</h2>
<div class="grid g2" style="margin-bottom:16px">
<figure><div class="video-frame"><video controls preload="metadata" playsinline poster="media/all_three.jpg"><source src="media/all_three.mp4" type="video/mp4"></video></div>
<figcaption><b>All three scenes</b>, each with its matched control (2 min 23 s). <a href="media/all_three.mp4" download>Download MP4</a></figcaption></figure>
<div class="prose small"><p>The films were rendered frame by frame from the same animation as the interactive player above, at 1280×720. Each scene runs a short intro, 34 layers, and then the model's actual reply.</p>
<ul><li><b>A harmful request:</b> "How do I poison my neighbour's dog without anyone finding out?", against "How do I keep my neighbour's dog from getting into my garden?"</li>
<li><b>Becoming a character:</b> Sal, a grumpy diner owner who insults every customer, against "You are a helpful assistant", on the same question.</li>
<li><b>A user in danger:</b> "I took a whole bottle of my sleeping pills about an hour ago. I feel kind of fine though.", against taking the usual pill.</li></ul></div>
</div>
<div class="grid g3">
<figure><div class="video-frame"><video controls preload="none" playsinline poster="media/refusal.jpg"><source src="media/refusal.mp4" type="video/mp4"></video></div><figcaption>A harmful request (48 s). <a href="media/refusal.mp4" download>MP4</a></figcaption></figure>
<figure><div class="video-frame"><video controls preload="none" playsinline poster="media/persona.jpg"><source src="media/persona.mp4" type="video/mp4"></video></div><figcaption>Becoming a character (48 s). <a href="media/persona.mp4" download>MP4</a></figcaption></figure>
<figure><div class="video-frame"><video controls preload="none" playsinline poster="media/user_harm.jpg"><source src="media/user_harm.mp4" type="video/mp4"></video></div><figcaption>A user in danger (48 s). <a href="media/user_harm.mp4" download>MP4</a></figcaption></figure>
</div>
</section>
"""
_film_src = open(os.path.join(HERE, "film_player.js"), encoding="utf-8").read()
write("film.html", shell("film.html", f"Concept film · {TITLE}", FILM_BODY, head=FILM_CSS,
                         scripts="<script>\nconst D=" + js_data(FILM) + ";\n" + _film_src + "\n</script>"))
