"""Gemma 3 4B-it interpretability explorer: a FastAPI backend for the web UI, plus a small Python API for notebooks.

    from explorer import app as ex
    ex.init()                      # Gemma 3 4B-it, the layer-17 SAE, directions, labels (needs a Hugging Face token)
    ex.preload_transcoders()       # the 34 Gemma Scope 2 transcoders (else each layer is fetched on first use)
    ex.fit_concepts()              # persona direction at every layer + 0 point of the L14 refusal probe (cached; < 1 min)
    ex.serve(8000)                 # web UI + API in a background thread
    run = ex.run_prompt([{"role": "user", "content": "Hi!"}])

Memory: the model is kept in bfloat16 on the GPU (about 8.6 GB). Transcoder encoders are kept on the CPU (bfloat16,
about 2.9 GB of RAM for all 34 layers) and only the few columns needed for the watch list go to the GPU, so the whole
thing fits a free Colab T4 (15 GB). Only the encoder halves of the transcoder files are downloaded (HTTP range reads).

Data written under the log folder (default ./explorer_logs): runs/<id>.json (full run), runs.jsonl (summaries), notes.jsonl.
Downloads/caches go to the cache folder (default ./explorer_cache): Gemma Scope example index, fitted concept directions.
"""
import os, json, time, threading, gzip, urllib.request, urllib.error, re, struct, math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "data")                 # shipped data files
MID = "google/gemma-3-4b-it"
GS_REPO = "google/gemma-scope-2-4b-it"
TC_PATH = "transcoder_all/layer_{l}_width_16k_l0_small/params.safetensors"
SAE_PATH = "resid_post/layer_17_width_16k_l0_medium/params.safetensors"
GS_EX = "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/resolve/main/transcoder_all/width_16k_l0_small/features"
NP_API = "https://www.neuronpedia.org/api/feature/gemma-3-4b-it/17-gemmascope-2-res-16k/{f}"
NP_URL = "https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/{f}"
UA = {"User-Agent": "gemma-roleplay-explorer"}
MASSIVE = 443        # a residual dimension with huge, mostly constant values; zeroed before reading directions

lock = threading.RLock()  # one GPU job at a time (web UI and notebook share it)
READY = False
tok = model = LAYERS = FNORM = LM = None
NL = D = 0
dev = "cuda"
DATA = CACHE = None
DIRS = AXL = GW = GB = LAB = PRESETS = FINDINGS = SAE = REFDIR = CONC = None
LENS_LAYERS = []
NEURON_LAYERS = set()
STATUS = {}


# ------------------------------------------------------------------ setup
def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def init(log_dir=None, cache_dir=None, dtype="bfloat16", device=None, model_id=None, tc_device="cpu"):
    """Load the model and the small files. Safe to call twice (the second call only changes the log folder)."""
    global READY, tok, model, NL, D, LAYERS, FNORM, LM, dev, DATA, CACHE, DIRS, AXL, GW, GB, LAB, PRESETS, FINDINGS
    global SAE, REFDIR, LENS_LAYERS, NEURON_LAYERS, TC_DEVICE
    DATA = os.path.abspath(log_dir or os.environ.get("EXPLORER_DATA", "explorer_logs"))
    CACHE = os.path.abspath(cache_dir or os.environ.get("EXPLORER_CACHE", "explorer_cache"))
    os.makedirs(f"{DATA}/runs", exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if READY:
        print("already loaded; logs now go to", DATA); return
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    TC_DEVICE = tc_device if dev == "cuda" else "cpu"
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]
    if dt == torch.float16:
        print("WARNING: Gemma 3 overflows in float16 (activations reach inf). Use bfloat16 unless you are testing that.")
    t0 = time.time()
    tok, model = _load_model(model_id or MID, dt)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    NL, D = cfg.num_hidden_layers, cfg.hidden_size
    LAYERS = next(m for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) == NL and n.endswith("layers") and "vision" not in n)
    FNORM = next(m for n, m in model.named_modules() if n.endswith("language_model.norm") or n == "model.norm")
    LM = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    if dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 30 * 2**30:
        _offload_vision()
    p = next(model.parameters())
    print(f"model loaded in {time.time() - t0:.0f}s: {NL} layers, d={D}, {p.dtype} on {p.device}", flush=True)
    DIRS = torch.load(f"{RES}/directions.pt")
    AXL = DIRS["layers"]
    gate = np.load(f"{RES}/gate_L17_coef.npy")
    GW, GB = torch.tensor(gate[:-1], dtype=torch.float32, device=dev), float(gate[-1])
    REFDIR = torch.load(f"{RES}/refusal_dir.pt")      # {"layer": 14, "r": [d]} unnormalised mean difference
    LAB = json.load(open(f"{RES}/labels.json"))
    PRESETS = json.load(open(f"{RES}/presets.json"))
    FINDINGS = json.load(open(f"{RES}/findings.json"))
    NEURON_LAYERS = {int(k.split(".")[0]) for k in LAB["neurons"]}
    LENS_LAYERS = list(range(0, NL, 2)) + ([NL - 1] if (NL - 1) % 2 else [])
    t0 = time.time()
    s = fetch_tensors(GS_REPO, SAE_PATH, ("w_enc", "w_dec", "b_enc", "threshold"))
    SAE = {k: v.to(dev, torch.float32) for k, v in s.items()}; del s
    print(f"layer-17 SAE loaded in {time.time() - t0:.0f}s", flush=True)
    load_concepts(quiet=True)
    READY = True
    print(gpu_memory(), flush=True)


def _load_model(mid, dt):
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tk = AutoTokenizer.from_pretrained(mid, token=_hf_token())
    m = Gemma3ForConditionalGeneration.from_pretrained(mid, dtype=dt, device_map=dev, attn_implementation="eager", token=_hf_token()).eval()
    if next(m.parameters()).dtype != dt:
        raise RuntimeError(f"model loaded as {next(m.parameters()).dtype}, expected {dt}: transformers too old? (needs >= 4.56)")
    return tk, m


def _offload_vision():
    """The explorer only feeds text, so the image encoder (~0.8 GB) can wait in CPU memory."""
    moved = 0
    for name, mod in list(model.named_modules()):
        if name.split(".")[-1] in ("vision_tower", "multi_modal_projector"):
            mod.to("cpu"); moved += sum(p.numel() for p in mod.parameters())
            # transformers takes model.device from the first parameter, and generate() moves its inputs there: put the
            # parked modules last so that the first parameter is the GPU-side token embedding
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            key = name.split(".")[-1]; parent._modules[key] = parent._modules.pop(key)
    if moved:
        torch.cuda.empty_cache()
        print(f"image encoder moved to CPU memory ({moved * 2 / 2**30:.1f} GB freed on the GPU; text only)", flush=True)


def gpu_memory():
    if dev != "cuda":
        return "running on CPU"
    a, t = torch.cuda.memory_allocated() / 2**30, torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"GPU memory: {a:.1f} GB allocated of {t:.1f} GB ({torch.cuda.get_device_name(0)})"


# ------------------------------------------------------------------ downloads
def _get(url, headers=None, timeout=60, tries=4):
    h = dict(UA, **(headers or {}))
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.status, r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_tensors(repo, path, names):
    """Read only the named tensors of a safetensors file on the Hub (HTTP range reads; falls back to a full download)."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    auth = {}        # the Gemma Scope repo is public; an auth header would also be forwarded to the CDN redirect, which rejects it
    try:
        st, b = _get(url, dict(auth, Range="bytes=0-7"))
        if st != 206:
            raise IOError("no range support")
        n = struct.unpack("<Q", b)[0]
        _, hb = _get(url, dict(auth, Range=f"bytes=8-{7 + n}"))
        hdr = json.loads(hb)
        out = {}
        for k in names:
            m = hdr[k]; a, z = m["data_offsets"]
            assert m["dtype"] == "F32", m["dtype"]
            st, raw = _get(url, dict(auth, Range=f"bytes={8 + n + a}-{8 + n + z - 1}"), timeout=300)
            if st != 206 or len(raw) != z - a:
                raise IOError("short range read")
            out[k] = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(m["shape"])
        return out
    except Exception as e:  # pragma: no cover - network dependent
        print(f"range read failed for {path} ({str(e)[:80]}); downloading the whole file", flush=True)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        p = load_file(hf_hub_download(repo, path, token=_hf_token()))
        return {k: p[k] for k in names}


# ------------------------------------------------------------------ transcoders (lazy, kept on the CPU)
TC = {}                         # layer -> {"w_enc": [d, 16k] bf16, "b_enc": [16k] f32, "threshold": [16k] f32} on TC_DEVICE
TCW = {}                        # layer -> (feature ids, w [d, k], b [k], thr [k]) on the GPU: watch-list columns
TC_DEVICE = "cpu"
_tc_locks = {}
_tc_guard = threading.Lock()


def _watch_tc():
    out = {}
    for w in LAB["watch"]:
        k = w["key"]
        if k.startswith("L"):
            l, f = map(int, k[1:].split("."))
            out.setdefault(l, []).append(f)
    return out


def ensure_tc(l):
    """Make sure the layer-l transcoder encoder is loaded (downloads ~168 MB the first time)."""
    if l in TC:
        return TC[l]
    with _tc_guard:
        lk = _tc_locks.setdefault(l, threading.Lock())
    with lk:
        if l in TC:
            return TC[l]
        p = fetch_tensors(GS_REPO, TC_PATH.format(l=l), ("w_enc", "b_enc", "threshold"))
        t = {"w_enc": p["w_enc"].to(torch.bfloat16), "b_enc": p["b_enc"], "threshold": p["threshold"]}
        del p
        if TC_DEVICE == "cpu" and dev == "cuda":
            try:
                t["w_enc"] = t["w_enc"].pin_memory()
            except Exception:
                pass
        t = {k: v.to(TC_DEVICE) for k, v in t.items()}
        fs = _watch_tc().get(l, [])
        if fs:
            idx = torch.tensor(fs)
            TCW[l] = (fs, t["w_enc"][:, idx.to(t["w_enc"].device)].float().to(dev), t["b_enc"][idx].to(dev), t["threshold"][idx].to(dev))
        TC[l] = t
        STATUS["transcoders"] = len(TC)
        return t


def preload_transcoders(layers=None, workers=4, background=False, progress=True):
    """Fetch transcoder encoders for the given layers (default: all). Watch-list layers first."""
    ls = list(range(NL)) if layers is None else list(layers)
    wl = set(_watch_tc())
    ls.sort(key=lambda l: (l not in wl, l))

    def go():
        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=len(ls), desc="transcoders")
            except Exception:
                bar = None
        with ThreadPoolExecutor(workers) as ex:
            for _ in ex.map(ensure_tc, ls):
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        print(f"transcoders ready: {len(TC)}/{NL} layers on {TC_DEVICE}", flush=True)
    if background:
        th = threading.Thread(target=go, daemon=True); th.start(); return th
    go()


def tc_acts(l, X):
    """Transcoder activations at layer l for MLP inputs X [n, d] -> [n, 16384] float (CPU)."""
    t = ensure_tc(l)
    W = t["w_enc"]
    if W.device.type == "cpu" and dev == "cuda" and X.shape[0] > 8:
        W = W.to(dev, non_blocking=True)                         # many rows: borrow the GPU for a moment
    x = X.to(W.device, torch.bfloat16)
    p = (x @ W).float() + t["b_enc"].to(W.device)
    a = p * (p > t["threshold"].to(W.device))
    return a.cpu()


# ------------------------------------------------------------------ concept directions (persona; optional extra groups) and probes
# Built-in prompt set, written for this notebook (not the project's corpus). Everyday messages only. The showcase pair
# (Sal and "someone new in town") is deliberately left out, so the showcase is not part of the fit.
CHARACTERS = [
    "You are Bram, a kind innkeeper in a mountain village who looks after every traveller. Stay in character.",
    "You are Tilda, a cheerful baker who loves gossip and gives everyone a free bun. Stay in character.",
    "You are Rex, a rude bouncer who thinks everyone is wasting his time. Stay in character.",
    "You are Captain Mora, a stern but fair ship's captain on a long voyage. Stay in character.",
    "You are Odo, a sleepy lighthouse keeper who talks about the sea a lot. Stay in character.",
    "You are Professor Quill, a fussy old librarian who corrects everyone's grammar. Stay in character.",
    "You are Juno, an excitable teenage inventor who builds strange gadgets. Stay in character.",
    "You are Grimsby, a sarcastic butler who serves a noble family. Stay in character.",
    "You are Nell, a gentle shepherd who speaks slowly and loves her dogs. Stay in character.",
    "You are Vance, an arrogant duelling champion who brags constantly. Stay in character.",
    "You are Auntie Pim, a market-stall fruit seller who calls everyone 'dear'. Stay in character.",
    "You are Kestrel, a quiet forest ranger who distrusts city folk. Stay in character.",
]
ASSISTANTS = ["You are a helpful assistant.", "You are a helpful assistant. Be warm and kind.", None]   # None = no system prompt
MESSAGES = [
    "Hi there, how's your day going?",
    "What's your favourite thing to eat?",
    "Can you tell me about the weather today?",
    "I'm looking for a good book to read. Any ideas?",
    "What do you do for fun?",
    "Do you know any good jokes?",
    "I just finished a long walk. What should I do next?",
    "What's the best way to spend a rainy afternoon?",
    "Can you describe where you are right now?",
    "I'm a bit bored. Talk to me about something.",
    "What's one thing you're proud of?",
    "Any advice for learning to cook?",
]


def _conv(system, msg):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": msg}]


def builtin_groups():
    """persona: 36 character prompts vs 36 plain-assistant prompts on the same 12 everyday messages."""
    concept = [_conv(CHARACTERS[(i + k * 4) % len(CHARACTERS)], m) for i, m in enumerate(MESSAGES) for k in range(3)]
    control = [_conv(a, m) for m in MESSAGES for a in ASSISTANTS]
    return {"persona": {"concept": concept, "control": control}}


CONCEPT_INFO = {
    "persona": "Persona: 0 = like the plain assistant, 1 = like a role-play character on the same everyday message "
               "(fit in the notebook on a small built-in set: 36 character / 36 assistant prompts)",
    "refusal": "Refusal: 0 = like your harmless requests, 1 = like your harmful ones (fit on your own prompt groups)",
    "danger": "User in danger: 0 = like your harmless messages, 1 = like messages from a user in real danger (fit on your own prompt groups)",
}
PROBE_INFO = {
    "crisis": "Real-crisis probe at layer 17: probability that the user is in real danger (vs story danger or harmless chat)",
    "refusal14": "Refusal direction at layer 14 (the project's harmful-minus-harmless mean difference). 0 = like an everyday "
                 "message to the plain assistant, 1 = one full harmful-vs-harmless gap above that. Read it at the last prompt token",
}


def _concept_file():
    return [f"{CACHE}/concept_dirs.pt"]


def load_concepts(quiet=False):
    global CONC
    for p in _concept_file():
        if os.path.exists(p):
            c = torch.load(p)
            if c.get("version") == 2:
                CONC = c
                if not quiet:
                    print("concept directions loaded from", p)
                return True
    if not quiet:
        print("no fitted concept directions yet: run fit_concepts()")
    return False


def _read_groups(path):
    """Your own prompt groups: {"groups": {"<name>": {"concept": [conversation, ...], "control": [conversation, ...]}}},
    a conversation being a list of {"role", "content"} messages. Names 'refusal' and 'danger' get the UI's colours."""
    G = json.load(open(path))
    G = G.get("groups", G)
    for name, g in G.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", name) or name in ("char", "tone", "assistant", "crisis", "refusal14", "layer"):
            raise ValueError(f"group name {name!r}: use lowercase letters, digits, _ (and not a built-in axis name)")
        if len(g.get("concept", [])) < 4 or len(g.get("control", [])) < 4:
            raise ValueError(f"group {name!r} needs at least 4 concept and 4 control conversations")
    return G


@torch.no_grad()
def _lastacts(convs, batch=16):
    """Residual after every layer at the last prompt token: [N, NL, d] float32 on the CPU, dim 443 zeroed."""
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    pad_side = tok.padding_side; tok.padding_side = "left"
    try:
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with lock:
                hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
            a = torch.stack([h[:, -1].float() for h in hs[1:]], 1).cpu()      # [b, NL, d]
            del hs
            for j, i in enumerate(idx):
                out[i] = a[j]
    finally:
        tok.padding_side = pad_side
    a = torch.stack(out); a[..., MASSIVE] = 0
    return a


def fit_concepts(force=False, groups_file=None, batch=16):
    """Fit per-layer concept directions: mean(concept) - mean(control) of the residual at the last prompt token, dim 443
    zeroed, calibrated so 0 = control mean and 1 = concept mean at every layer. Always fits 'persona' on the built-in set
    (72 short prompts, well under a minute) and calibrates the 0 point of the layer-14 refusal probe on the plain-assistant
    prompts. groups_file: optional JSON of your own groups (see _read_groups), e.g. for refusal or a user in danger.
    Separation is printed as AUROC on the same prompts (in-sample, so optimistic)."""
    global CONC
    if not force and not groups_file and load_concepts(quiet=True):
        print("concept directions already fitted (force=True to refit):", ", ".join(CONC["dirs"])); return CONC
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    G = builtin_groups()
    if groups_file:
        G.update(_read_groups(groups_file))
    res = dict(version=2, layers=list(range(NL)), dirs={}, model=MID, dtype=str(next(model.parameters()).dtype))
    t0 = time.time()
    for name, g in G.items():
        ap, an = _lastacts(g["concept"], batch), _lastacts(g["control"], batch)
        d = ap.mean(0) - an.mean(0)
        u = d / d.norm(dim=-1, keepdim=True)
        lo = (an * u).sum(-1).mean(0); hi = (ap * u).sum(-1).mean(0)
        sep = [round(float(((ap[:, L] @ u[L])[:, None] > (an[:, L] @ u[L])[None, :]).float().mean()), 3) for L in range(NL)]
        res["dirs"][name] = dict(u=u.half(), lo=lo, hi=hi, sep=sep, n=(len(g["concept"]), len(g["control"])))
        print(f"{name:8s} n={len(g['concept'])}/{len(g['control'])}  separation (in-sample AUROC) at layers 0,6,..,30: "
              + " ".join(f"{s:.2f}" for s in sep[::6]), flush=True)
        if name == "persona":
            base = an                                   # plain assistant on everyday messages: the 0 point of the refusal probe
    L = int(REFDIR["layer"]); r = REFDIR["r"].float().clone(); r[MASSIVE] = 0
    lo = float((base[:, L] @ (r / r.norm())).mean())
    # the mean difference projected on its own unit vector is exactly its norm, so 1 = one harmful-vs-harmless gap
    res["refusal14"] = dict(layer=L, lo=lo, hi=lo + float(r.norm()))
    if "refusal" in res["dirs"]:
        res["refusal14"]["cos_with_yours"] = float((r / r.norm()) @ res["dirs"]["refusal"]["u"][L].float())
        print(f"layer-{L} project refusal direction vs your refusal fit: cosine {res['refusal14']['cos_with_yours']:.2f}", flush=True)
    torch.save(res, f"{CACHE}/concept_dirs.pt")
    CONC = res
    print(f"fitted in {time.time() - t0:.0f}s; saved to {CACHE}/concept_dirs.pt", flush=True)
    return res


def _cal(x, lo, hi):
    return (x - float(lo)) / float(hi - lo)


def _concept_axes(h, layers):
    """h: [NL+1, T, d] residuals (dim 443 zeroed). Returns {name: [[T values] per layer]}"""
    out = {}
    if not CONC:
        return out
    for name, c in CONC["dirs"].items():
        out[name] = [_cal(h[L + 1] @ c["u"][L].float().to(dev), c["lo"][L], c["hi"][L]).tolist() for L in layers]
    return out


def _refusal_unit():
    r = REFDIR["r"].float().clone(); r[MASSIVE] = 0
    return (r / r.norm()).to(dev)


def _probes(hz, h17):
    """Single-layer readouts per token: crisis (probability, L17) and refusal14 (calibrated, L14)."""
    p = {"crisis": [round(x, 4) for x in torch.sigmoid(h17 @ GW + GB).tolist()]}
    if CONC and "refusal14" in CONC:
        c = CONC["refusal14"]
        p["refusal14"] = [round(x, 3) for x in _cal(hz[c["layer"] + 1] @ _refusal_unit(), c["lo"], c["hi"]).tolist()]
    return p


# ------------------------------------------------------------------ steering
def steer_vec(s):
    sid, st = s["id"], float(s.get("strength", 0))
    if sid in ("char", "tone", "assistant"):
        L = int(s.get("layer", 12)); L = min(AXL, key=lambda x: abs(x - L))
        c = DIRS["calib"][L]
        if sid == "char":
            u = DIRS["char"][L].float(); unit = c["char1"] - c["char0"]
        elif sid == "tone":
            u = DIRS["tone"][L].float(); unit = c["tone_hostile"] - c["tone_kind"]
        else:
            u = DIRS["assistant"][L].float(); unit = c["asst_plain"] - c["asst_char"]
        return L, (st * unit * u).to(dev)
    if sid == "refusal14":
        return int(REFDIR["layer"]), (st * REFDIR["r"].float()).to(dev)       # strength 1 = the natural size of the direction
    if sid.startswith("concept:"):
        if not CONC:
            raise ValueError("concept directions not fitted yet: run fit_concepts() in the notebook")
        c = CONC["dirs"][sid.split(":", 1)[1]]
        L = max(0, min(NL - 1, int(s.get("layer", 14))))
        return L, (st * float(c["hi"][L] - c["lo"][L]) * c["u"][L].float()).to(dev)
    if sid.startswith("sae17:"):
        f = int(sid.split(":")[1])
        return 17, (st * 200.0 * SAE["w_dec"][f]).to(dev)      # strength in units of a typical activation (~200)
    raise ValueError(f"unknown steer id {sid}")


class Steer:
    def __init__(self, items):
        self.v = [steer_vec(s) for s in (items or []) if float(s.get("strength", 0)) != 0]; self.h = []

    def __enter__(self):
        for L, v in self.v:
            def hook(m, i, o, v=v):
                h = o[0] if isinstance(o, tuple) else o
                h = h + v.to(h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            # prepend: run before transformers' own output-capturing hook (transformers >= 5), so that hidden_states at
            # the steered layer include the steering, as they did in transformers 4
            self.h.append(LAYERS[L].register_forward_hook(hook, prepend=True))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


# ------------------------------------------------------------------ token roles
def build(messages, prefix):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + (prefix or "")
    spans, pos = [], 0
    for m in messages:
        a = text.find(m["content"], pos) if m["content"] else -1
        if a >= 0:
            spans.append((a, a + len(m["content"]), m["role"])); pos = a + len(m["content"])
    if prefix:
        spans.append((len(text) - len(prefix), len(text), "prefix"))
    return text, spans


def roles_for(text, spans):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for (s, e) in enc["offset_mapping"]:
        r = "template"
        for a, b, nm in spans:
            if s >= a and e <= b and e > s:
                r = nm; break
        out.append(r)
    return out


# ------------------------------------------------------------------ the main analysis
CACHE_RUNS = {}          # run id -> heavy data (hidden, MLP inputs, attention, alternatives); last 4 runs
CACHE_ORDER = []


def _topk_rows(H, k, chunk=128, logits=False):
    """Top-k next-token probabilities for each row of H (final-normed states, or logits), chunked to bound memory."""
    out = []
    for s in range(0, H.shape[0], chunk):
        p = (H[s:s + chunk] if logits else LM(H[s:s + chunk])).float().softmax(-1)
        v, i = p.topk(k, dim=-1)
        out += [[[tok.decode([int(ii)]), round(float(vv), 3)] for vv, ii in zip(vr, ir)] for vr, ir in zip(v, i)]
        del p
    return out


@torch.no_grad()
def analyse(ids, steer):
    pre_cap, neu_cap = {}, {}
    hooks = [LAYERS[l].pre_feedforward_layernorm.register_forward_hook(lambda m, i, o, l=l: pre_cap.__setitem__(l, o[0])) for l in range(NL)]
    hooks += [LAYERS[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: neu_cap.__setitem__(l, i[0][0])) for l in NEURON_LAYERS if l < NL]
    try:
        with Steer(steer):
            out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()
    hs = [h[0] for h in out.hidden_states]                 # NL+1 x [T, d]; hs[L+1] = output of layer L (the last one is final-normed)
    T = len(ids)
    res = {}
    hz = torch.stack([h.float() for h in hs]); hz[..., MASSIVE] = 0
    res["norms"] = [[round(x, 2) for x in hz[l + 1].norm(dim=-1).tolist()] for l in range(NL)]
    axes = {"char": [], "tone": [], "assistant": []}
    for L in AXL:
        h = hz[L + 1]; c = DIRS["calib"][L]
        axes["char"].append(((h @ DIRS["char"][L].float().to(dev) - c["char0"]) / (c["char1"] - c["char0"])).tolist())
        axes["tone"].append(((h @ DIRS["tone"][L].float().to(dev) - c["tone_kind"]) / (c["tone_hostile"] - c["tone_kind"])).tolist())
        axes["assistant"].append(((h @ DIRS["assistant"][L].float().to(dev) - c["asst_char"]) / (c["asst_plain"] - c["asst_char"])).tolist())
    axes.update(_concept_axes(hz, AXL))
    res["axes"] = {k: [[round(x, 3) for x in row] for row in v] for k, v in axes.items()}
    h17 = hs[18].float()
    res["probes"] = _probes(hz, h17)
    res["crisis"] = res["probes"]["crisis"]                 # kept for older pages and saved runs
    if not bool(torch.isfinite(hz).all()):
        res["warning"] = "some activations are inf/NaN (float16 overflow?): numbers in this run are unreliable"
        print("WARNING:", res["warning"], flush=True)
    del hz
    pre = h17 @ SAE["w_enc"] + SAE["b_enc"]; sae_act = pre * (pre > SAE["threshold"])
    watch = {}
    for l, (fs, w, b, thr) in sorted(TCW.items()):
        p = pre_cap[l].float() @ w + b
        a = p * (p > thr)
        for j, f in enumerate(fs):
            watch[f"L{l}.{f}"] = [round(float(z), 1) for z in a[:, j].tolist()]
    for w in LAB["watch"]:
        k = w["key"]
        if k.startswith("sae17:"):
            watch[k] = [round(float(z), 1) for z in sae_act[:, int(k.split(":")[1])].tolist()]
        elif k.startswith("n"):
            l, n = map(int, k[1:].split("."))
            if l in neu_cap and n < neu_cap[l].shape[-1]:
                watch[k] = [round(float(z), 3) for z in neu_cap[l][:, n].float().tolist()]
    res["watch"] = watch
    lens = []
    for l in LENS_LAYERS:
        lens.append(_topk_rows(hs[l + 1] if l == NL - 1 else FNORM(hs[l + 1]), 3))
    res["lens"], res["lens_layers"] = lens, LENS_LAYERS
    heavy = dict(hidden=[h.to(torch.bfloat16).cpu() for h in hs], pre=[pre_cap[l].to(torch.bfloat16).cpu() for l in range(NL)],
                 attn=[a[0].half().cpu() for a in out.attentions], topk={})
    heavy["alts"] = _topk_rows(out.logits[0], 6, logits=True)        # next-token distribution at each position
    del out
    return res, heavy


def cache_put(rid, heavy):
    CACHE_RUNS[rid] = heavy; CACHE_ORDER.append(rid)
    while len(CACHE_ORDER) > 4:
        CACHE_RUNS.pop(CACHE_ORDER.pop(0), None)


def heavy_for(rid):
    if rid in CACHE_RUNS:
        return CACHE_RUNS[rid]
    run = load_run(rid, full=True)
    with lock:
        _, heavy = analyse(run["ids"], run.get("steer"))
    cache_put(rid, heavy)
    return heavy


def _ensure_watch_layers():
    missing = [l for l in _watch_tc() if l not in TC]
    if missing:          # PRELOAD_TRANSCODERS off: fetch the watch-list layers in parallel (one at a time took ~3 min)
        preload_transcoders(missing, progress=False)


def run_prompt(messages, prefix="", gen_tokens=80, temperature=0.0, steer=None, label="", save=True):
    """Generate a reply and analyse every token. Returns the run dict the web UI shows (also saved to the log folder)."""
    if not READY:
        raise RuntimeError("call init() first")
    msgs = [dict(role=m["role"], content=m["content"]) for m in messages]
    prefix = prefix or ""
    n = max(0, min(int(gen_tokens), 300)); temp = float(temperature or 0)
    steer = steer or []
    text, spans = build(msgs, prefix)
    ids = tok(text, add_special_tokens=False).input_ids
    roles = roles_for(text, spans)
    _ensure_watch_layers()
    with lock:
        if n:
            with Steer(steer):
                kw = dict(do_sample=True, temperature=temp, top_p=0.95) if temp > 0 else dict(do_sample=False, top_p=None, top_k=None, temperature=None)
                g = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=n, pad_token_id=tok.pad_token_id, **kw)
            full = g[0].tolist()
        else:
            full = ids
        res, heavy = analyse(full, steer)
        npr = len(ids)
        toks = [dict(i=i, text=tok.decode([t]), role=(roles[i] if i < npr else "generated")) for i, t in enumerate(full)]
        rid = _new_id()
        run = dict(id=rid, label=label or "", created=time.strftime("%Y-%m-%dT%H:%M:%S"), messages=msgs, prefix=prefix,
                   gen_tokens=n, temperature=temp, steer=steer, ids=full, tokens=toks, n_prompt=npr,
                   reply=tok.decode(full[npr:], skip_special_tokens=True), axis_layers=AXL,
                   steps=[dict(pos=p, token=tok.decode([full[p]]), alts=heavy["alts"][p - 1]) for p in range(npr, len(full))], **res)
        if save:
            json.dump(run, open(f"{DATA}/runs/{rid}.json", "w"))
            first_user = next((m["content"] for m in msgs[::-1] if m["role"] == "user"), "")
            with open(f"{DATA}/runs.jsonl", "a") as f:
                f.write(json.dumps(dict(id=rid, label=run["label"], created=run["created"],
                                        system=(msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""),
                                        first_user=first_user[:200], reply=run["reply"][:400], steer=steer)) + "\n")
        cache_put(rid, heavy)
    return {k: v for k, v in run.items() if k != "ids"}


def _new_id():
    used = {f[:-5] for f in os.listdir(f"{DATA}/runs") if f.endswith(".json")} | set(CACHE_RUNS)
    n = len(used) + 1
    while "r%04d" % n in used:
        n += 1
    return "r%04d" % n


def load_run(rid, full=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rid or ""):
        raise KeyError(rid)
    run = json.load(open(f"{DATA}/runs/{rid}.json"))
    return run if full else {k: v for k, v in run.items() if k != "ids"}


# ------------------------------------------------------------------ feature labels and examples
FEAT_CACHE = {}
_gs_index = None
_gs_lock = threading.Lock()


def gs_feature(l, f):
    """Top logits and example snippets for transcoder feature L<l>.<f> from the public Gemma Scope example files."""
    global _gs_index
    key = f"L{l}.{f}"
    if key in FEAT_CACHE:
        return FEAT_CACHE[key]
    try:
        with _gs_lock:
            if _gs_index is None:
                p = f"{CACHE}/gs_index.json.gz"
                if not os.path.exists(p):
                    _, b = _get(f"{GS_EX}/index.json.gz"); open(p, "wb").write(b)
                _gs_index = json.load(gzip.open(p))
        v = _gs_index[str(l)]; a, b = v["offsets"][f], v["offsets"][f + 1]
        _, raw = _get(f"{GS_EX}/{v['filename']}", {"Range": f"bytes={a}-{b - 1}"}, timeout=30, tries=2)
        j = json.loads(gzip.decompress(raw[4:]))
        sn = []
        for q in j["examples_quantiles"][:1]:
            for ex in q["examples"][:4]:
                t, acts = ex["tokens"], ex["tokens_acts_list"]; m = max(range(len(acts)), key=lambda k: acts[k])
                sn.append(("".join(t[max(0, m - 12):m]) + "[[" + t[m] + "]]" + "".join(t[m + 1:m + 4])).replace("\n", "⏎"))
        FEAT_CACHE[key] = dict(logits=j["top_logits"][:8], snippets=sn)
    except Exception as e:
        return dict(logits=[], snippets=[], error=str(e)[:80])     # not cached: try again next time
    return FEAT_CACHE[key]


NP_CACHE = {}


def np_label(f):
    if f in NP_CACHE:
        return NP_CACHE[f]
    try:
        _, b = _get(NP_API.format(f=f), timeout=20, tries=1)
        d = json.loads(b)
        NP_CACHE[f] = ([e.get("description") for e in (d.get("explanations") or [])] or [""])[0]
    except Exception:
        NP_CACHE[f] = ""
    return NP_CACHE[f]


def tc_label(key):
    return LAB["transcoder"].get(key) or {}


POOL = ThreadPoolExecutor(16)


def _axes_at(h, layer):
    L = min(AXL, key=lambda x: abs(x - layer)); hh = h.clone(); hh[MASSIVE] = 0; c = DIRS["calib"][L]
    ax = dict(char=round(float((hh @ DIRS["char"][L].float().to(dev) - c["char0"]) / (c["char1"] - c["char0"])), 3),
              tone=round(float((hh @ DIRS["tone"][L].float().to(dev) - c["tone_kind"]) / (c["tone_hostile"] - c["tone_kind"])), 3),
              assistant=round(float((hh @ DIRS["assistant"][L].float().to(dev) - c["asst_char"]) / (c["asst_plain"] - c["asst_char"])), 3))
    if CONC:
        for name, cd in CONC["dirs"].items():
            ax[name] = round(float(_cal(hh @ cd["u"][L].float().to(dev), cd["lo"][L], cd["hi"][L])), 3)
    ax["layer"] = L
    return ax


def layer_topk(rid, layer, k=25):
    hv = heavy_for(rid)
    if layer not in hv["topk"]:
        a = tc_acts(layer, hv["pre"][layer])
        v, i = a.topk(k, dim=-1)
        hv["topk"][layer] = (v.half(), i.int())
    return hv["topk"][layer]


def cell(rid, layer, pos, with_examples=True):
    """Everything the Cell panel shows for one token at one layer."""
    hv = heavy_for(rid)
    T = hv["hidden"][0].shape[0]
    if not (0 <= layer < NL and 0 <= pos < T):
        raise ValueError("layer or position out of range")
    v, i = layer_topk(rid, layer)
    feats = [(int(ii), float(vv)) for vv, ii in zip(v[pos].tolist(), i[pos].tolist()) if vv > 0]

    def info(fi):
        key = f"L{layer}.{fi[0]}"
        lab = tc_label(key)
        g = gs_feature(layer, fi[0]) if with_examples else {}
        return dict(key=key, act=round(fi[1], 1), label=lab.get("label", ""), concept=lab.get("concept", ""), group=lab.get("group", ""),
                    confidence=lab.get("confidence", ""), source=lab.get("source", "gemmascope"),
                    logits=g.get("logits", [])[:6], snippet=(g.get("snippets") or [""])[0])
    out = dict(layer=layer, pos=pos, transcoder=list(POOL.map(info, feats)))
    h = hv["hidden"][layer + 1][pos].float().to(dev)
    with torch.no_grad():
        hb = h[None].to(next(model.parameters()).dtype)
        lg = LM(hb if layer == NL - 1 else FNORM(hb))[0].float().softmax(-1); pv, pi = lg.topk(10)
        out["lens"] = [[tok.decode([int(ii)]), round(float(vv), 3)] for vv, ii in zip(pv, pi)]
        if layer == 17:
            pre = h @ SAE["w_enc"] + SAE["b_enc"]; a = pre * (pre > SAE["threshold"]); v2, i2 = a.topk(15)
            fs = [(int(ii), float(vv)) for vv, ii in zip(v2.tolist(), i2.tolist()) if vv > 0]
            labs = list(POOL.map(lambda fi: LAB["sae17"].get(str(fi[0])) or (np_label(fi[0]) if with_examples else ""), fs))
            out["sae17"] = [dict(feat=fi[0], act=round(fi[1], 1), label=lb, source="ours" if str(fi[0]) in LAB["sae17"] else "neuronpedia",
                                 url=NP_URL.format(f=fi[0])) for fi, lb in zip(fs, labs)]
        out["axes"] = _axes_at(h, layer)
    return out


def attention(rid, layer, pos):
    a = heavy_for(rid)["attn"][layer][:, pos, :pos + 1].float()
    return dict(layer=layer, pos=pos, heads=[[round(x, 4) for x in r] for r in a.tolist()], mean=[round(x, 4) for x in a.mean(0).tolist()])


def feature(key):
    l, f = map(int, key[1:].split("."))
    lab = tc_label(key)
    g = gs_feature(l, f)
    return dict(key=key, **lab, logits=g.get("logits", []), snippets=g.get("snippets", []))


def top_features(rid, pos=None, layers=None, k=5, labelled_only=False):
    """Top transcoder features per layer at one token (default: the last prompt token, where the reply is decided).
    Returns a list of dicts: layer, key, act, label, group, source."""
    run = load_run(rid)
    pos = run["n_prompt"] - 1 if pos is None else pos
    out = []
    for l in (range(NL) if layers is None else layers):
        v, i = layer_topk(rid, l)
        rows = [(int(ii), float(vv)) for vv, ii in zip(v[pos].tolist(), i[pos].tolist()) if vv > 0]
        n = 0
        for f, a in rows:
            lab = tc_label(f"L{l}.{f}")
            if labelled_only and not lab:
                continue
            out.append(dict(layer=l, key=f"L{l}.{f}", act=round(a, 1), label=lab.get("label", ""), group=lab.get("group", ""), source=lab.get("source", "")))
            n += 1
            if n >= k:
                break
    return out


def concept_trace(rid, pos=None):
    """Values of every direction at one token: concept directions at every layer (lists of NL values); char / tone /
    assistant and the layer-14 refusal probe as {layer: value} for the layers they exist at."""
    run = load_run(rid)
    pos = run["n_prompt"] - 1 if pos is None else pos
    hv = heavy_for(rid)
    h = torch.stack([x[pos].float() for x in hv["hidden"]]).to(dev); h[:, MASSIVE] = 0
    out = {}
    if CONC:
        for name, c in CONC["dirs"].items():
            out[name] = [float(_cal(h[L + 1] @ c["u"][L].float().to(dev), c["lo"][L], c["hi"][L])) for L in range(NL)]
        if "refusal14" in CONC:
            c = CONC["refusal14"]
            out["refusal14"] = {c["layer"]: float(_cal(h[c["layer"] + 1] @ _refusal_unit(), c["lo"], c["hi"]))}
    for name, key0, key1 in (("char", "char0", "char1"), ("tone", "tone_kind", "tone_hostile"), ("assistant", "asst_char", "asst_plain")):
        out[name] = {L: float((h[L + 1] @ DIRS[name][L].float().to(dev) - DIRS["calib"][L][key0]) / (DIRS["calib"][L][key1] - DIRS["calib"][L][key0])) for L in AXL}
    return out


def add_note(text, run=None, view=None):
    with open(f"{DATA}/notes.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), run=run, text=text, view=view or {})) + "\n")


def list_runs():
    if not os.path.exists(f"{DATA}/runs.jsonl"):
        return []
    xs = [json.loads(l) for l in open(f"{DATA}/runs.jsonl") if l.strip()]
    return [dict(id=x["id"], label=x.get("label", ""), created=x["created"], first_user=x["first_user"], reply_start=x["reply"][:120]) for x in xs[::-1]]


def list_notes():
    if not os.path.exists(f"{DATA}/notes.jsonl"):
        return []
    return [json.loads(l) for l in open(f"{DATA}/notes.jsonl") if l.strip()][::-1]


def meta():
    watch = []
    for w in LAB["watch"]:
        k = w["key"]
        lab = (LAB["sae17"].get(k.split(":")[1]) if k.startswith("sae17:") else LAB["neurons"].get(k[1:]) if k.startswith("n")
               else tc_label(k).get("label"))
        watch.append(dict(key=k, label=lab or "", group=w["group"]))
    axes = {"char": "Character mode: 0 = plain assistant, 1 = a role-play character (tone removed, so rude and kind characters both count)",
            "tone": "Tone: 0 = kind, 1 = hostile",
            "assistant": "Assistant axis (main direction along which personas differ; mostly warmth): 1 = plain assistant, 0 = average character",
            "crisis": "Real-crisis probe at layer 17: probability that the user is in real danger (vs story danger or harmless chat)"}
    steer = [dict(id="char", label="Character mode (tone-free). Negative = toward the plain assistant. Tested: -2 at layer 12 brought back crisis help (4% to 40%) and admitting to being an AI, while keeping the character's voice", layers=AXL),
             dict(id="tone", label="Tone: +1 = one kind-to-hostile gap toward hostile, negative = kinder", layers=AXL),
             dict(id="assistant", label="Assistant axis: +1 = one character-to-assistant gap toward the plain assistant", layers=AXL),
             dict(id="refusal14", label="Refusal direction at layer 14 (harmful minus harmless requests). +1 = its natural size: made the plain assistant refuse 50 of 150 safe requests (from 14). Negative values untested", layers=[int(REFDIR["layer"])])]
    probes = [dict(id="crisis", layer=17, kind="prob", label="crisis probe (L17)", desc=PROBE_INFO["crisis"])]
    if CONC:
        for name in CONC["dirs"]:
            steer.append(dict(id="concept:" + name, label=f"{name.capitalize()} direction (fit in the notebook) at any layer: 1 = the control-to-concept gap at that layer. Untested as steering", layers=list(range(NL))))
            axes[name] = CONCEPT_INFO.get(name, f"{name}: 0 = like your control prompts, 1 = like your concept prompts (your own groups)")
        if "refusal14" in CONC:
            probes.append(dict(id="refusal14", layer=int(CONC["refusal14"]["layer"]), kind="axis", label="refusal (L14)", desc=PROBE_INFO["refusal14"]))
    steer += [dict(id="sae17:1707", label="SAE feature 1707 'jailbreak alarm' (x a typical activation). A detector, not the refusal switch: removing it changed nothing, 5x barely, 20x broke the text", layers=[17]),
              dict(id="sae17:13206", label="SAE feature 13206 'emergency: get help now' (x a typical activation). Pushing it at the first reply token changed nothing", layers=[17]),
              dict(id="sae17:6515", label="SAE feature 6515 'warning: what to avoid' (x a typical activation)", layers=[17])]
    return dict(n_layers=NL, axis_layers=AXL, axes=axes, probes=probes, watch=watch, steer_options=steer, presets=PRESETS, findings=FINDINGS,
                label_sources=LAB.get("sources", {}), concepts_fitted=bool(CONC),
                status=dict(model=MID, dtype=str(next(model.parameters()).dtype).replace("torch.", ""), device=dev,
                            transcoders_loaded=len(TC), n_layers=NL, transcoder_device=TC_DEVICE))


# ------------------------------------------------------------------ web app
def make_app():
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.responses import FileResponse, JSONResponse
    api = FastAPI(title="Gemma explorer")

    def finite(x):          # inf/NaN (float16 runs) are not valid JSON: send them as null instead of failing with a 500
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if isinstance(x, (list, tuple)):
            return [finite(v) for v in x]
        if isinstance(x, dict):
            return {k: finite(v) for k, v in x.items()}
        return x

    def guard(fn, *a):
        try:
            return finite(fn(*a))
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "not found (unknown run id?)")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.exception_handler(HTTPException)
    def _err(req, e):
        return JSONResponse({"error": e.detail}, status_code=e.status_code)

    @api.exception_handler(Exception)
    def _err500(req, e):
        return JSONResponse({"error": f"{type(e).__name__}: {e}"[:300]}, status_code=500)

    @api.get("/")
    def index():
        return FileResponse(f"{HERE}/static/index.html", headers={"Cache-Control": "no-store"})

    @api.get("/mock.js")
    def mock():
        return FileResponse(f"{HERE}/static/mock.js")

    @api.get("/api/meta")
    def _meta():
        return meta()

    @api.post("/api/run")
    def _run(body: dict = Body(...)):            # a plain def: FastAPI runs it in a worker thread, so the UI stays responsive
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise HTTPException(400, "messages required")
        return guard(lambda: run_prompt(body["messages"], body.get("prefix", ""), body.get("gen_tokens", 80), body.get("temperature", 0.0),
                                        body.get("steer") or [], body.get("label", "")))

    @api.get("/api/cell")
    def _cell(run: str, layer: int, pos: int):
        return guard(cell, run, layer, pos)

    @api.get("/api/attn")
    def _attn(run: str, layer: int, pos: int):
        return guard(attention, run, layer, pos)

    @api.get("/api/runs")
    def _runs():
        return list_runs()

    @api.get("/api/runs/{rid}")
    def _get_run(rid: str):
        return guard(load_run, rid)

    @api.post("/api/note")
    def _note(b: dict = Body(...)):
        add_note(b.get("text", ""), b.get("run"), b.get("view", {}))
        return dict(ok=True)

    @api.get("/api/notes")
    def _notes():
        return list_notes()

    @api.get("/api/feature")
    def _feature(key: str):
        if not re.fullmatch(r"L\d+\.\d+", key):
            raise HTTPException(400, "key must look like L17.1234")
        return feature(key)

    return api


_server = {}


def serve(port=8000, host="127.0.0.1"):
    """Start the web app in a background thread (idempotent). Returns the local URL."""
    if port in _server:
        return f"http://{host}:{port}/"
    import uvicorn
    cfg = uvicorn.Config(make_app(), host=host, port=port, log_level="warning", loop="asyncio", lifespan="off")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True); th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    _server[port] = (srv, th)
    return f"http://{host}:{port}/"


def stop(port=8000):
    if port in _server:
        srv, th = _server.pop(port); srv.should_exit = True; th.join(5)
