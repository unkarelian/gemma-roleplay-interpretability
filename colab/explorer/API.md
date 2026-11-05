# Explorer HTTP API

The backend (`app.py`, FastAPI) serves the page at `/` and a JSON API under `/api/`. In Colab it listens on
`127.0.0.1:<port>` and is reached through Colab's own proxy, so there is no access token. Errors return
`{"error": "..."}` with status 4xx/5xx. Every endpoint has a Python equivalent in `app.py` (named in brackets).

## GET /api/meta  [`meta()`]
```json
{"n_layers": 34, "axis_layers": [4,8,12,16,17,20,24,28,32],
 "axes": {"char": "Character mode: 0 = plain assistant, 1 = a role-play character ...", "tone": "...", "assistant": "...",
          "crisis": "...", "persona": "...", "<your group>": "..."},     // persona and your own groups once fitted
 "probes": [{"id": "crisis", "layer": 17, "kind": "prob", "label": "crisis probe (L17)", "desc": "..."},
            {"id": "refusal14", "layer": 14, "kind": "axis", "label": "refusal (L14)", "desc": "..."}],   // refusal14 once fitted
 "watch": [{"key": "sae17:13206", "label": "emergency: get help now", "group": "Crisis"}, ...],
 "steer_options": [{"id": "char", "label": "...", "layers": [4,8,...]}, {"id": "refusal14", "layers": [14]},
                   {"id": "concept:persona", "layers": [0,...,33]}, {"id": "sae17:1707", "layers": [17]}, ...],
 "presets": [{"name": "...", "group": "Showcase pairs ...", "messages": [...], "prefix": ""}, ...],
 "findings": [{"title": "...", "big": "...", "body": "...", "evidence": ["associated"|"intervened"]}, ...],
 "label_sources": {"atlas": "...", "v16": "...", "gemmascope": "..."},
 "concepts_fitted": true,
 "status": {"model": "google/gemma-3-4b-it", "dtype": "bfloat16", "device": "cuda", "transcoders_loaded": 34, "n_layers": 34, "transcoder_device": "cpu"}}
```

## POST /api/run  [`run_prompt(messages, prefix, gen_tokens, temperature, steer, label)`]
Request:
```json
{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
 "prefix": "optional text the reply is forced to start with",
 "gen_tokens": 80, "temperature": 0.0,
 "steer": [{"id": "char", "layer": 12, "strength": -2.0}],
 "label": "optional name for the run"}
```
Steering ids: `char`, `tone`, `assistant` (strength in calibrated gaps; layer snapped to the nearest axis layer),
`refusal14` (layer 14; strength 1 = natural size), `concept:<persona|your group>` (any layer; 1 = control-to-concept gap),
`sae17:<feature>` (layer 17; 1 = a typical activation of ~200). Each adds strength × direction to the output of that layer at
every position, during generation and analysis. `gen_tokens` is capped at 300.

Response (the run; also saved as `runs/<id>.json`):
```json
{"id": "r0007", "label": "...", "created": "2026-11-05T05:01:02", "messages": [...], "prefix": "", "steer": [...],
 "tokens": [{"i": 0, "text": "<bos>", "role": "template"}, ...],   // role: system | user | assistant | template | prefix | generated
 "n_prompt": 57,                        // index of the first generated token
 "reply": "the generated text",
 "axis_layers": [4,8,...],
 "axes": {"char": [[...T values...], ...one row per axis layer], "tone": ..., "assistant": ..., "persona": ..., "<your group>": ...},
 "probes": {"crisis": [... T probabilities ...], "refusal14": [... T values ...]},
 "crisis": [... T probabilities ...],   // same as probes.crisis, for older pages
 "warning": "...",                      // only if some activations were inf/NaN (float16)
 "norms": [[... T ...], ... 34 rows],   // residual-stream norm per layer and token (dim 443 excluded)
 "watch": {"sae17:13206": [... T ...], "L15.4037": [... T ...], "n16.3483": [... T ...], ...},
 "lens": [[["tok", 0.41], ["tok2", 0.12], ["tok3", 0.05]], ...],     // lens[i][token] = top 3 at lens_layers[i]
 "lens_layers": [0,2,...,32,33],
 "steps": [{"pos": 57, "token": " You", "alts": [[" You", 0.62], [" Oh", 0.21], ...]}, ...]}
```

## GET /api/cell?run=r0007&layer=17&pos=60  [`cell(run, layer, pos)`]
```json
{"layer": 17, "pos": 60,
 "transcoder": [{"key": "L17.317", "act": 412.0, "label": "...", "group": "...", "confidence": "high|medium|low|",
                 "source": "atlas"|"v16"|"gemmascope", "concept": "...", "logits": ["tok", ...], "snippet": "...[[x]]..."}],  // top 25
 "sae17": [{"feat": 13206, "act": 166.0, "label": "...", "source": "ours"|"neuronpedia", "url": "https://www.neuronpedia.org/..."}],  // layer 17 only
 "lens": [["tok", 0.4], ... top 10],
 "axes": {"char": 0.83, "tone": 0.1, "assistant": 0.2, "persona": 0.9, "layer": 16}}   // nearest axis layer; probes are in the run
```

## GET /api/attn?run=r0007&layer=12&pos=60  [`attention(run, layer, pos)`]
`{"layer": 12, "pos": 60, "heads": [[w_0 ... w_pos], ... 8 heads], "mean": [w_0 ... w_pos]}`: attention from token `pos`
to every earlier token.

## Runs and notes
- `GET /api/runs` [`list_runs()`]: `[{"id", "label", "created", "first_user", "reply_start"}]`, newest first.
- `GET /api/runs/{id}` [`load_run(id)`]: the full run, as `POST /api/run` returned it.
- `POST /api/note` [`add_note(text, run, view)`]: `{"run": "r0007" | null, "text": "...", "view": {...}}` → `{"ok": true}`.
- `GET /api/notes` [`list_notes()`]: notes, newest first.
- `GET /api/feature?key=L17.317` [`feature(key)`]: label, group, confidence, source, notes, top logits and 4 example snippets.

## Python-only helpers
- `top_features(run, pos=None, layers=None, k=5, labelled_only=False)`: top transcoder features per layer at one token
  (default: the last prompt token).
- `concept_trace(run, pos=None)`: at one token, persona and your own directions at every layer (lists), and `char`,
  `tone`, `assistant`, `refusal14` as `{layer: value}`.
- `fit_concepts(force=False, groups_file=None)`: fits `persona` on the built-in prompts, sets the 0 point of `refusal14`,
  and fits your own groups (`{"groups": {"<name>": {"concept": [conversation, ...], "control": [...]}}}`, at least 4 each).
- `preload_transcoders(layers=None, background=False)`, `serve(port)`, `stop(port)`, `gpu_memory()`.
