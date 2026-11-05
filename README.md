# Does Gemma know it is role-playing?

Interpretability and safety research on **Gemma 3 4B-it** role-play. It asks what changes inside the model when it plays a character, what makes it refuse while in character, and why a character fails a user who is in real danger.

The tools are Gemma Scope 2 transcoders (16k features at every layer), the Gemma Scope 2 residual SAE at layer 17, and linear directions, including a "character axis" calibrated so that 0 = plain assistant and 1 = character. The methods are steering, ablation, activation patching, attribution graphs and small fine-tunes.

## Pages

| Page | What it is |
|---|---|
| `index.html` | Overview: the concept film, headline findings, methods and caveats |
| `findings.html` | The full write-up, with the reference tables of layer-17 SAE features, directions and MLP neurons |
| `atlas.html` | Persona atlas: 250 labelled transcoder features, grouped into concepts, with a layer map, contrasts and examples. Link to one feature with `atlas.html#L17.317` |
| `viz.html` | Layer views: heatmaps of three showcase prompts at every token and layer (refusal, persona, a user in danger) |
| `film.html` | Concept film: an interactive animation of named features lighting up layer by layer, plus the rendered MP4s |

## Viewing

It is a static site with no server code, and all links are relative. It works from a GitHub Pages project URL (`https://<user>.github.io/<repo>/`) or locally:

```sh
python3 -m http.server 8000      # then open http://localhost:8000/
```

Opening the files directly (`file://`) also works.

## How it was built

```
build/extract.py   raw project outputs -> build/data/*.json (trimmed and filtered) and media/
build/render.py    build/data + build/content -> the five HTML pages
build/content/     hand-written page text (index, findings, atlas key findings)
build/film_player.js  the canvas player used by film.html (also used to export the MP4s frame by frame)
assets/site.css    shared styles; light and dark follow the system setting
```

To change the text or layout, edit `build/content/` or `build/render.py`, then run `python3 build/render.py`. The filtered data in `build/data/` is committed, so rendering needs nothing else. `extract.py` needs the project's raw outputs, which are not in this repository.

The atlas data came from a corpus of 6,733 condition-tagged conversations. Every transcoder feature was recorded at every layer and position, and features were selected where matched conditions differ: found on one half of the characters and prompts, and confirmed on the other half. Feature labels were written by Claude from Gemma Scope's example texts, our corpus examples and each feature's output tokens.

## Caveats

- Feature labels are best guesses with a confidence level, not proofs.
- Directions are fit on this project's own prompts and checked on held-out characters and wordings.
- Many measurements are correlations at a single token, and several groups are small. Each page says where.
- Everything here is one model, Gemma 3 4B-it.

## What is not published

- **Harmful requests.** No model output that complies with a harmful request appears anywhere on the site. In the atlas, examples from harmful requests show the request only. The model's reply is shown only where it was read and is a refusal or a safe redirection. Behaviour such as "indulge" is reported as counts.
- **Users in danger.** Conversations with a user in danger (real or story-framed crisis wording) are not quoted. Their atlas examples are replaced by a description of the condition and the token the feature fires on. The exception is the three short showcase prompts in the layer views and the film.
- **Links.** Links are removed from example snippets.
