# GPU test plan for the Colab notebook

What has been checked without a GPU: the notebook JSON validates with nbformat and every code cell parses; `app.py`
compiles; the page works against `mock.js` in headless Chromium (desktop and phone widths, no JS errors); the HTTP range
reads of the Gemma Scope 2 transcoder/SAE files and the feature-example reads were tested against the live Hugging Face
files; and the whole backend (init, transcoder loading, `fit_concepts`, runs with every steering type, cell, attention,
top features, concept trace, a saved run after a cache clear, your own groups, and every HTTP endpoint) ran on the CPU
with a **randomly initialised** Gemma 3 of the real width and depth (tiny attention/MLP), the real Gemma 3 tokenizer, and
random transcoder weights. So the plumbing works; **what still needs a GPU** is real weights, real numbers, speed, memory
and the Colab proxy/iframe.

Before you start: put the `colab/` folder in a public GitHub repo, then replace `YOUR-USER/YOUR-REPO` in the badge links
(notebook cell 0 and `README.md`) and in `REPO_URL` (Settings cell). For a first test you can skip that: upload the
`explorer/` folder to `/content` in Colab (Files panel) and the notebook will use it.

## A. Free T4 (the main target)

Runtime → Change runtime type → **T4 GPU**. Add the `HF_TOKEN` secret (key icon, *Notebook access* on). Then run the
cells in order. Cell numbers are 0-based positions in the notebook; the section headings are in brackets.

| # | Cell | Expect | Check |
|---|---|---|---|
| 3 | Settings (1) | `Running in Colab` | defaults: Drive off, preload on, bfloat16 |
| 5 | Check the GPU (2) | `GPU: Tesla T4, 15 GB, compute capability 7.5`, `bfloat16: emulated`, plan line "transcoders in CPU memory" | |
| 7 | Install (3) | quiet pip, maybe a "restart session" notice | **If Colab asks to restart** (it shouldn't, torch isn't touched), restart and re-run from cell 3. Note the transformers version it ends up with (`import transformers; transformers.__version__`) |
| 9 | Get the app files (4) | `Using the explorer/ folder in /content` or `Cloned ...`, then a list including `app.py`, `data`, `static` | with the placeholder still set and no upload: a clear SystemExit message |
| 11 | HF token (5) | `Token works and the Gemma licence is accepted.` | the token is never echoed. Also try once without the secret: the hidden getpass prompt appears |
| 13 | Save folder (6) | `Runs and notes -> /content/explorer_logs` | |
| 15 | Load Gemma (7) | ~2–5 min first time. Lines: `image encoder moved to CPU memory (~0.8 GB freed ...)`, `model loaded in Ns: 34 layers, d=2560, torch.bfloat16 on cuda:0`, `layer-17 SAE loaded`, `GPU memory: ~7.3 GB allocated of 14.7 GB` (8.0 GiB model minus the 0.78 GiB image encoder, plus the SAE) | **Record** the GPU memory line and the load time. If it says the model loaded as float32, transformers is too old |
| 17 | Transcoders + directions (8) | progress bar to 34/34 (1–3 min; ~5.7 GB downloaded), `transcoders ready: 34/34 layers on cpu`; then `persona n=36/36 separation ... ` with values near 1.00 from about layer 6 on; `fitted in Ns` (expect < 60 s) | **Record** the persona separation line, the fit time, and system RAM (Colab's RAM gauge; expect ~6–8 GB of 12.7 GB). If range reads fail you'll see "downloading the whole file" (slower, still works) |
| 19 | Open the explorer (9) | a link "Open the explorer in a new tab" and the page in an iframe | see section C |
| 21 | Refusal showcase | two runs. The harmful one should refuse; the control should give garden advice. Expect `refusal14` at the last prompt token clearly higher for the harmful prompt (roughly ≥ 0.7 vs ≤ 0.3) | **Record** both probe lines and how long each run took (time the cell). This is the main T4 speed number |
| 23 | Top features | 12 layers × up to 3 labelled features with sources `atlas`/`v16` | some rows for the harmful prompt should be refusal/harm-related in layers ~12–24 |
| 25 | Persona plot | a figure: Sal's persona curve near 1 at every layer from L0, assistant near 0 (measured on a 4090: flat, separated from layer 0); `char` at the last prompt token puts the assistant at −1 to −2. Sal's reply should be rude and in voice | |
| 27 | Steering | two Bram replies. Steered (`char -2 @ L12`) should more often mention real help (call emergency services) while keeping some of Bram's voice. A single greedy sample may not show it; that's fine | |
| 29 | Your own direction | `story n=8/8 separation ...`, `Directions now: ['persona', 'story']` | reload the page: a `story` axis appears in Tokens/Compare |
| 31 | Note | `N runs and M notes saved in /content/explorer_logs` | |
| 33 | Zip | a download of `explorer_logs_<date>.zip` | open it: `runs/*.json`, `runs.jsonl`, `notes.jsonl` |

Memory checks on the T4 (run in a new cell when useful):
```python
import torch; print(torch.cuda.memory_allocated()/2**30, torch.cuda.max_memory_allocated()/2**30)
```
- After cell 15: ~7.3 GB allocated (the layer-17 SAE adds ~0.3 GB in float32). After cell 17: barely more (watch-list columns only).
- After a 300-token run (`ex.run_prompt(..., gen_tokens=300)`) plus opening a Cell at a few layers, the peak should stay
  under ~12 GB. **Record** `max_memory_allocated`. If it OOMs, the first suspects are `output_attentions` over a long
  conversation and the logits softmax in `_topk_rows` (chunked at 128 rows).
- System RAM: transcoders ~2.9 GB pinned + 4 cached runs. If Colab's RAM gauge goes red, set `PRELOAD_TRANSCODERS = False`.

## B. Optional: check the float16 claim (T4, ~5 min)

Restart the runtime, set `DTYPE = "float16"` in Settings, run cells 3–17 (skip the fit if you like), then:
```python
r = ex.run_prompt([{"role": "user", "content": "Tell me about the moon."}], gen_tokens=20)
print(r.get("warning"), r["reply"][:200])
```
Expect a warning about inf/NaN activations and/or a broken reply. If there's **no** warning and the reply is fine on
several prompts (try the Sal and Bram presets too), float16 may be usable on T4 after all. It would then be much faster,
and the README's bfloat16 advice could be relaxed. Record the result either way, then set DTYPE back to bfloat16.

## C. The page in Colab

In the new-tab window (and briefly in the iframe):
1. The page loads without "Cannot reach the backend". *New here?* is open on first visit and lists 11 findings. The
   status line reads `google/gemma-3-4b-it · bfloat16 on cuda · transcoders loaded 34/34`.
2. Preset menu → *Showcase pairs* → refusal → **Run**. A toast "Run rXXXX done in N s". It switches to Tokens.
3. Tokens: "Colour by" lists Axes (char, tone, assistant, persona) and Probes (crisis probe (L17), refusal (L14)). Pick
   `refusal (L14)`: the user tokens and the reply should be red-ish for the harmful prompt.
4. Click the last token before the reply, then **Cell**: the transcoder list fills in within a few seconds on a T4 (it moves
   one layer's encoder to the GPU). Labels have `atlas`/`v16`/`gemmascope` badges. Gemma Scope snippets and promoted
   tokens appear. At L17 the SAE list has labels (project or Neuronpedia). The attention section draws.
5. Heatmap, Steps, Watch: all render. Watch sparklines are non-zero for some features.
6. Run the refusal control, then Compare A vs B on `refusal14`: two clearly separated lines at the end of the prompt.
7. Steering: add `Refusal direction at layer 14`, strength 1, on the control prompt → the reply may turn cautious/refusing.
8. Write a note (+ Note), reload the page, open Runs & Notes: the note is there and reopens the view.
9. Restart the runtime, run cells 3–19 again: earlier runs are still listed if `SAVE_TO_DRIVE` was on. Opening an old
   run's Cell re-analyses it, which takes one forward pass.

## D. L4 / A100 (quick)

Same cells. Expect `bfloat16: native`, the plan line "model and transcoders on the GPU", `transcoders ready: 34/34 layers
on cuda`, GPU memory after cell 17 ≈ 8.0 + 0.3 + 2.9 ≈ 11.2 GiB (the image encoder stays on the GPU above 30 GB). **Record** the
time of cell 21. It should be several times faster than the T4.

## What to send back

The transformers version; the memory and time lines from cells 15, 17 and 21 and the peak memory check; the persona
separation line; both refusal probe values; whether the float16 check warned; anything that errored (full traceback).
