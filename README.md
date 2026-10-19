# Handwriting Synthesis

A from-scratch implementation of [Graves (2013) — *Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850), trained on the IAM On-Line Handwriting Database and served as a streaming HTTP API.

You give it text. It generates realistic cursive handwriting, stroke by stroke.

---

## How it works

The model uses a **soft attention window** to align pen strokes with input characters. Three stacked LSTMs generate a mixture-density output (20 Gaussians) that models the joint distribution of pen offsets and pen-up events.

```
text: "hello world"
         ↓
   soft window (K=10 Gaussians along text axis)
         ↓
   3× stacked LSTM  (hidden size 400)
         ↓
   mixture density network (M=20)
         ↓
   pen strokes (dx, dy, pen_up)
```

The window is monotonic — κ only moves forward through the text — so the model can't cheat by re-reading characters.

---

## Samples

Temperature controls how "messy" the handwriting is (0 = deterministic, 1 = expressive).

| temp 0.0 | temp 0.3 | temp 0.5 | temp 1.0 |
|----------|----------|----------|----------|
| clean, rigid | slightly looser | natural | chaotic |

![Temperature comparison](docs/temp_comparison.png)

The numpy serving backend produces bit-for-bit identical output to the original PyTorch model:

![Torch vs numpy parity](docs/torch_vs_numpy.png)

---

## API

Live at: `https://handwriting-api-687921010800.europe-central2.run.app`

### Stream strokes (recommended)

```bash
curl -X POST https://handwriting-api-687921010800.europe-central2.run.app/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "temperature": 0.5}'
```

Returns NDJSON — one `{"type": "point", "x": ..., "y": ..., "pen_up": 0|1}` per pen step, streamed live. Use this to animate the handwriting as it generates rather than waiting for the full result.

### JSON response

```bash
curl -X POST https://handwriting-api-687921010800.europe-central2.run.app/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "temperature": 0.5}'
```

### Other endpoints

| `GET /vocab` | list of drawable characters (77 total) |
|---|---|
| `GET /health` | liveness check + backend info |

Polish characters (ą, ę, ó, …) are automatically transliterated to ASCII. Anything else outside the vocab returns `400`.

---

## Architecture

| Component | Detail |
|---|---|
| Model | 3× LSTM, H=400, soft window K=10, MDN M=20 |
| Training | IAM On-Line Handwriting DB, 80 epochs, batch 80 |
| Checkpoint | `models/best-4.pth` is the base model. Other `.pth` files are experiments. |
| Serving | Pure numpy (no torch in production) |
| Deployment | Google Cloud Run, `europe-central2`, scale-to-zero |
| Image size | ~98 MB (vs ~334 MB with CPU torch) |
| Cold start | ~0.1–0.5s (numpy import; torch was 5–10s) |

The serving backend is a hand-ported numpy reimplementation of the forward pass (`model_np.py`), validated to < 2×10⁻⁵ absolute error against torch. This eliminates the `import torch` cold start entirely — the biggest user-perceptible latency was the import, not the generation.

---

## Running locally

```bash
# install deps (uv recommended)
uv sync

# generate a sample (torch)
uv run generate.py

# run the numpy API server
uv run app_np.py
# → http://127.0.0.1:5000

# generate a comparison grid
uv run python scripts/grid_samples.py
```

Weights live on Hugging Face ([PanzerBread/handwriting](https://huggingface.co/PanzerBread/handwriting)), not in this repo. Fetch them before running:

```bash
uv pip install "huggingface_hub[cli]"
hf download PanzerBread/handwriting weights.npz stoi.json std.json --local-dir .
```

### Training

```bash
# needs the IAM On-Line Handwriting Dataset in data/
uv run train.py
# saves checkpoints to models/post-trained.pth

# export for serving
uv run export_weights.py
# models/post-trained.pth → weights.npz + stoi.json
```

`models/best-4.pth` is the base model used for evaluation and for the post-training stages below. Checkpoints written by `train.py` are experiments.

### Post-training

Both stages keep `models/best-4.pth` as a frozen KL anchor and never write that file. The update is sequence-level REINFORCE on the stroke log-probability: length-normalized, with a per-prompt baseline, and a KL penalty back toward the base model. `train_rl.py` starts from `models/rl-trocr.pth` when that file exists, and from the base model otherwise.

**Recognizer reward.** TrOCR-Large reads a rendering of each sample. The default score is plain CER plus an exact-match bonus. A lookalike mistake costs the same as any other edit. `--reward vcer` is the older visual distance (`docs/vcer.md`).

| `--reward` | value |
|---|---|
| `cer` (default) | `0.8 * (1 - min(CER, 1)) + 0.2 * exact`, then mixed with TrOCR confidence (`--confidence`, default 0.2). High confidence raises the reward. |
| `vcer` | The same shape, with a visual edit distance in place of CER. |
| `exp` | `exp(-CER)`, in (0, 1] |
| `linear` | `1 - CER` |
| `soft` | `1` when CER is at or below `--tau` (default 0.1), then a straight fall to 0 at CER = 1 |

A crooked line then loses a little more. Two parallel lines sit on the top and the bottom of the ink and share one slope. A flat pair, slope 0 and no arch, loses nothing. `--straightness` (default 0.05) is the most that penalty can subtract.

Prompts default to `data/romeo_juliet.txt`, the whole play cut into lines of at most 48 characters and sampled at random. `--prompts data/sentences.txt` is the IAM pool. `--max-chars` defaults to 48 so TrOCR's square resize does not squash the line. On `--novel` of draws (default 0.25) the line is three random words of letters and digits from the model alphabet, such as `aeGyAcs OIxwy sskN1ewuyS`, so the policy also practices strings the play does not contain. `--novel 0` keeps the file. `--text` stays on that one sentence.

One prompt is grown a word at a time. The first word is drawn `--beam` times (default 9) from a blank page. The best `--keep` (default 3) continue, and each is extended `beam / keep` ways. TrOCR scores the ink of the whole prefix against the words written so far, including the word just added. The gradient covers only the new strokes, compared with the other continuations of the same prefix. One logged update is the whole line. The charts average every drawing in that line. The sample view has a slider for prefix length, and shows the 9 drawings of the length you pick. `-k` and `--batch` do not change this.

**Human preference.** The model draws 3 samples of one prompt. You pick the best. That sample scores 1 and the other two score 0.

```bash
uv sync --extra rl          # transformers, for the recognizer stage
uv run train_rl.py          # ~10 min of updates, then a checkpoint
                            # models/rl-trocr.pth is the latest; re-run resumes it
                            # samples/rl/ckpt-NNNN.png is 10 draws of one fixed sentence
                            # --ckpt samples/rl/ckpt-0003.pth continues from that one
uv run train_rlhf.py        # writes models/rl-human.pth, appends data/rlhf_prefs.jsonl
```

`train_rlhf.py` resumes from `models/rl-human.pth` if it exists, otherwise from `models/rl-trocr.pth`, otherwise from the base model. The KL anchor stays `models/best-4.pth`.

---

## Repo layout

```
model.py              # PyTorch model (training)
model_np.py           # numpy forward pass (serving)
app_np.py             # Flask API (deployed)
train.py              # supervised training loop
train_rl.py           # post-train on a TrOCR-Large CER reward
train_rlhf.py         # post-train on a human best-of-3 pick
rl.py                 # CER reward, trajectory log-prob, REINFORCE
generate.py           # sample from a checkpoint
export_weights.py     # .pth → weights.npz for serving
data.py               # dataset + dataloader
Dockerfile            # slim numpy image, no torch

models/               # checkpoints (gitignored)
  best-4.pth          #   base model, not overwritten by post-training
  rl-trocr.pth        #   recognizer-reward RL
  rl-human.pth        #   human best-of-3 RL

scripts/              # dev / demo tools
  grid_samples.py     #   sample grid across texts + temps
  generate_variants.py#   variant PNGs for video picking
  render_drawing_mp4.py#  animated MP4 from streamed inference
  import_collected.py #   canvas JSON → numpy
  export_dataset.py   #   merge sources → data/dataset.npz
  rebuild_dataset.py  #   rebuild with QA rejections
  iam_helper.py       #   review-tool helper

docs/                 # images, deployment notes, dev logs
```

---

## References

- Graves, A. (2013). [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
- [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
