# HANDOFF — run the training on your Mac

This repo is complete and verified: game engine, teacher solver, data pipeline, 0.2B transformer, training loop, evaluation, Gradio app, API benchmark harness, and all three notebooks. Everything runs. What's missing is a trained checkpoint — this machine (the Windows/WSL box) has no GPU, so the full 215M-parameter run never happened here. That's your Mac's job.

The plan: clone this repo on the MacBook Pro, run notebook 01 (train), notebook 02 (evaluate + charts), optionally notebook 03 (API benchmarks), then push results back. This file is the operator's manual. Read it once, then hand it to the Mac Hermes session.

## What's already done

- Game engine, entropy teacher (~99-100% solve, ~3.5 avg guesses), data generator with exploration and a 200-answer holdout
- 0.2B transformer (16 layers, 1024 hidden, ~215M params) with word-level tokenizer, masked word-position loss, MPS support
- All 3 notebooks patched for Apple Silicon (auto-picks mps; falls back to cpu)
- API benchmark harness (OpenAI / Anthropic / DeepSeek / Groq / Gemini / OpenRouter / local), `.env.example` with key names
- Gradio web app (play, hints, autoplay)
- `docs/improvement_walkthrough.md` — the playbook for pushing past the teacher's ceiling
- `docs/benchmarks.md` — the benchmark protocol
- `docs/explainer.md` — how the whole thing works under the hood

## What's NOT done (your Mac's job)

- Train the real 0.2B checkpoint (only a tiny smoke config has ever run)
- Run the holdout evaluation and generate real result charts
- Run the API model benchmark (needs API keys)
- Update the README with your actual numbers

## Machine requirements

- Apple Silicon (M1/M2/M3/M4, any Pro/Max/Ultra is better). Intel Macs work but fall back to CPU — slow.
- 16 GB RAM minimum. 32 GB comfortable.
- ~15 GB free disk (torch + data + checkpoints).
- Python 3.11 or 3.12. If you use Homebrew: `brew install python@3.12`.

## Setup (one time, ~5 minutes)

```bash
git clone https://github.com/AbdoDameen/wordle-0.2b.git
cd wordle-0.2b
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # includes torch with MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
# must print True. If False, you're on an Intel Mac or the wrong torch build.
```

Then verify the pipeline end-to-end (optional but smart, ~1 minute):

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from wordle02b import Vocabulary, load_word_lists, build_pattern_matrix_cached
from wordle02b.evaluate import split_answers, play_games_baseline, summarize
import numpy as np
vocab = Vocabulary(*load_word_lists())
P = build_pattern_matrix_cached(vocab, cache_dir='data/cache')
_, holdout = split_answers(vocab)
secrets = [vocab.word(int(w)) for w in np.random.default_rng(100).choice(holdout, 100, replace=False)]
print(summarize(play_games_baseline(vocab, P, 100, secrets, seed=100)))
"
# expect: solve_rate ~1.0, avg_guesses ~3.5. This is the teacher, your ceiling.
```

## Run order

### 1. Train — `notebooks/01_train.ipynb`

Run all cells. It will:

1. Build the pattern matrix (first run only, a few minutes, cached to `data/cache/`)
2. Generate 300,000 teacher games (~7M tokens, a few minutes with multiprocessing; cached)
3. Train the 215M model for 2,500 steps, logging loss every 50 and evaluating solve rate on the 200-word holdout every 250 steps
4. Save checkpoints every 500 steps to `checkpoints/`, final at `checkpoints/wordle-0.2b-final.pt`

Time: roughly 2-6 hours on an M1 Pro/Max-class machine at fp32. Watch the `eval step` lines: solve rate climbs and flattens. If it's flat at ≥95% before step 2,500, you can interrupt and keep the last checkpoint — `train.py` saves every 500 steps, nothing is lost.

Troubleshooting:
- Out of memory: in the config cell, set `BATCH = 256` (and rerun from the config cell). A 215M fp32 model is ~860MB of weights; activations at batch 512 are the bigger chunk.
- Wrong device: the notebook prints `device: mps` (or cuda/cpu). If it prints cpu on an M-series Mac, stop and fix the torch install.
- Want a fast sanity run first: set `SMOKE = True` in the config cell — tiny 6M model, 20k games, 600 steps, finishes in ~15-30 minutes, saves `wordle-smoke-final.pt`. Good dry run before the 2-6 hour commitment.

### 2. Evaluate + charts — `notebooks/02_play_and_evaluate.ipynb`

Runs 300 games on the holdout for the model vs the teacher, prints solve rate / avg guesses / guess distribution, saves `assets/solve_rate_comparison.png`, and gives you a playable widget (play yourself, hints, autoplay). The chart overwrites the placeholder I shipped — commit the real one.

Expected (fully trained): solve ≥95%, avg ≤3.9. The teacher sits at ~99-100% / ~3.5. If your model is meaningfully below the teacher, that's the "imitation tax" — see `docs/improvement_walkthrough.md`.

### 3. (Optional) API benchmarks — `notebooks/03_benchmark_api_models.ipynb` or CLI

The headline: a 0.2B model that beats frontier LLMs at Wordle. Same rules, same secrets, 6 guesses, single valid word per turn.

```bash
cp .env.example .env   # paste any API keys you have (all optional)
python benchmarks/api_benchmark.py --models \
  openai/gpt-4o-mini,anthropic/claude-sonnet-4-5,deepseek/deepseek-chat,baseline,local0.2b \
  --checkpoint checkpoints/wordle-0.2b-final.pt --games 30 --plot
```

No keys? The baseline and your model still run. Protocol details in `docs/benchmarks.md`.

### 4. (Optional) Web app

```bash
python app/gradio_app.py --checkpoint checkpoints/wordle-0.2b-final.pt
```

### 5. Push results back

```bash
git add assets/ results/ README.md docs/
git commit -m "trained 0.2B: <solve rate>% holdout, <avg> guesses (update with real numbers)"
git pull --rebase && git push
```

What to push: the real chart, any results JSON/CSV, and README updates (replace the "expected" table with actuals). What NOT to push: `checkpoints/`, `data/cache/`, `results/`, `.env` are gitignored on purpose.

The checkpoint is ~860MB — too big for the repo. If you want it public, upload it to HuggingFace (`pip install huggingface_hub`, `huggingface-cli upload <model_id> checkpoints/wordle-0.2b-final.pt`) or attach it to a GitHub Release (2GB limit). Update the README link if you do.

## For the Hermes session on the Mac

Give it this prompt, run from the repo root:

> Read HANDOFF.md and docs/explainer.md. Run notebooks/01_train.ipynb end to end (SMOKE=False). Confirm the device prints mps. Watch the eval solve rate; if it plateaus at >=95% early, stop and keep the latest checkpoint. Then run 02_play_and_evaluate.ipynb fully. Run the API benchmark CLI with whatever keys exist in .env (never print keys). Commit the chart, results, and README updates with real numbers and push. Report: final holdout solve rate, avg guesses, guess distribution, how the model compared to the teacher and to each API model.

If you're on a slow network or want to train overnight, run notebook 01 with `jupyter nbconvert --to notebook --execute --inplace notebooks/01_train.ipynb` in a terminal and let it grind.

## What to expect (sanity numbers)

| Contestant | Solve rate | Avg guesses |
|---|---|---|
| exact solver (upper bound) | ~100% | ~3.4 |
| entropy teacher | ~99% | ~3.5 |
| this model, fully trained | 95-98% | ~3.8 |
| gpt-4o-mini / deepseek-chat class | 70-90% | varies |
| strongest frontier chat models | 80-95% | varies |

Anything wildly outside these bands means something's wrong, not that you found a miracle. The model wins because it plays the game; the LLMs talk about it.
