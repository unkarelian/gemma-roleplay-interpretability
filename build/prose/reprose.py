"""Rewrite the site's prose with google/gemini-4-argon (OpenRouter), keeping markup and every number identical.
python reprose.py IN_DIR OUT_DIR  (IN_DIR holds chunk_*.txt; writes the same names to OUT_DIR, plus report.json)"""
import os, sys, re, json, glob, requests, concurrent.futures as cf
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
KEY = os.environ["OPENROUTER_API_KEY"]
SYS = """You are a copy editor for an interpretability research website. Rewrite the prose you are given in a dry, plain research register: precise, declarative, unadorned, like a careful methods-and-results section. No rhetorical questions, no dramatic framing, no "not X but Y" constructions, no colons used for reveals, no em-dash flourishes, no lists of three for rhythm, no summarising last sentences, no hedging filler, no second person unless the original instructs the reader to do something.

Hard constraints:
- Keep every HTML tag, attribute, link, id and class exactly as given and in the same order; change only the human-readable text between tags. For JSON input, keep keys and structure and rewrite only the string values of "title" and "body".
- Keep every number, percentage, ratio, layer/feature id (e.g. L17.1707, L14, 1707), model name, quoted prompt or quoted model output, and technical term exactly as written. Do not add, drop, round or reinterpret any result, claim, caveat or sample size.
- Keep roughly the same length or shorter. Do not add new content.
Return only the rewritten text, with no preamble and no code fences."""
TAG = re.compile(r"<[^>]+>")
NUM = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*%?|L\d+\.\d+")
def sig(t):
    return [m for m in TAG.findall(t)], sorted(NUM.findall(re.sub(r"<[^>]+>", " ", t)))
def call(text, extra=""):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=600,
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": "google/gemini-4-argon", "temperature": 0.3,
              "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text + extra}]})
    j = r.json(); out = j["choices"][0]["message"]["content"].strip()
    out = re.sub(r"^```\w*\n|\n```$", "", out)
    return out, j.get("usage", {})
def do(path):
    name = os.path.basename(path); src = open(path).read()
    t0, n0 = sig(src); note = ""
    for attempt in range(3):
        out, u = call(src, note)
        t1, n1 = sig(out)
        ok_json = True
        if name.endswith(".json.txt"):
            try:
                a, b = json.loads(src), json.loads(out)
                ok_json = len(a) == len(b) and all(set(x) == set(y) and all(x[k] == y[k] for k in x if k not in ("title", "body")) for x, y in zip(a, b))
            except Exception:
                ok_json = False
        if t0 == t1 and n0 == n1 and ok_json:
            open(os.path.join(OUT, name), "w").write(out)
            return name, "ok", attempt, u
        miss = sorted(set(n0) - set(n1)); add = sorted(set(n1) - set(n0))
        note = (f"\n\n[Your previous attempt changed the markup or numbers (numbers lost: {miss[:20]}, added: {add[:20]}, "
                f"tags identical: {t0 == t1}, json ok: {ok_json}). Redo it keeping them exactly.]")
    open(os.path.join(OUT, name), "w").write(src)
    return name, "KEPT ORIGINAL", attempt, u
with cf.ThreadPoolExecutor(8) as ex:
    rep = list(ex.map(do, sorted(glob.glob(os.path.join(IN, "chunk_*")))))
json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
for r in rep: print(r[:3])
print("REPROSE_DONE")
