# Explainer — how wordle-0.2b works

A ~215M-parameter transformer that learns to play Wordle from scratch by imitating a solver. No pretrained weights, no word2vec, no tricks. This document walks through every piece so you can debug it, explain it in an interview, or improve it without guessing.

## The game, as a sequence problem

Wordle is a conversation: guess a word, get 5 letters of feedback, guess again. The whole game state fits in one short sequence:

```
START  crane  GYBGB  slate  GGGYB  ...
```

Each guess is one token. Each feedback is 5 letters of feedback (G = green, Y = yellow, B = grey) as tokens. The model's job at every guess position: predict the next guess given everything so far. It's a language model over the language of Wordle games.

Why this framing works: the state is fully observable, the vocabulary is closed (12,972 legal words, 2,315 possible answers), and the rules are static. There's no hidden information and no long-horizon ambiguity. A small transformer with the right training data can represent near-optimal play in this space — it's a constrained decision problem, not open-ended language.

## The vocabulary

- Token 0: padding. Token 1: START.
- Tokens 2-4: the three feedback letters (G, Y, B).
- Tokens 5+: one token per legal word (12,972 total).

Word-level tokens are a deliberate trade: each guess is one token, so sequences are short (a full game is ~11 tokens per guess-pair, max 6 guesses = well under the block size of 40). The cost is that the model must learn letter structure from scratch inside 12,972 opaque embeddings. That's a lot of the difficulty — and a known upgrade path (letter-level tokenization) is documented in the improvement walkthrough.

## The teacher: a greedy entropy solver

`src/wordle02b/baseline.py`. Each turn it:

1. Keeps the set of candidate answers consistent with all feedback so far.
2. Scores candidate guesses by expected information: for each guess, how much does the average feedback pattern shrink the candidate set? Pick the guess with maximum expected entropy reduction.
3. Plays the best guess, filters candidates, repeats.

The full sweep over all 12,972 guesses every turn is expensive, so the solver samples a pool of ~300 candidates and picks the best within it. This is the classic Wordle heuristic; it wins ~99% of games in ~3.5 guesses, just short of the ~100% / ~3.4 of an exact decision-tree solver. That's the teacher, and the teacher's strength is the model's ceiling: a perfect imitator wins ~99%. Everything in the improvement walkthrough is about raising or bypassing that ceiling.

## The data: imitation with exploration

`src/wordle02b/data.py`. The teacher plays 300,000 games. Two things make the data teachable rather than memorizable:

1. **Exploration.** 80% of the time the teacher takes its top pick; 20% of the time it samples from its top-3. 30% of games start from a random word. Result: the same answer produces many different game trajectories, so the model sees diverse states instead of one path per answer.
2. **The holdout.** 200 of the 2,315 answers are split out of training entirely (`split_answers` in `evaluate.py`). All evaluation uses those 200. The model can never memorize the right answer for a holdout secret — it must generalize the skill.

Each game is serialized to the token stream (START, guess, feedback, guess, feedback...) and the whole corpus is one big array of ~7M tokens. Training samples random 40-token blocks from it.

## The model

A nanoGPT-style decoder-only transformer (`src/wordle02b/model.py`):

- 16 layers, 1024 hidden, 16 heads, ~215M parameters ("0.2B")
- RMSNorm (no LayerNorm), GELU MLP with 4x expansion, weight tying between embedding and output head
- No bias terms; GPT-2-style initialization with scaled residual projections

Nothing exotic. The architecture is deliberately boring because the task is the interesting part.

## The training objective

Cross-entropy over word tokens only. The loss is computed at every position whose target is a word token (a guess); feedback positions are masked out with `ignore_index=-1`. Rationale: the model should learn to produce the next guess, and nothing else. If it can also predict feedback, fine, but there's no gradient pressure forcing it to — and the mask keeps the model from spending capacity on trivial next-token prediction of feedback it already knows.

Training details: AdamW, cosine LR with 200-step warmup from 3e-4, weight decay 0.05, grad clip 1.0, batch 512 x 40 tokens. Every 250 steps the training loop pauses and plays 200 fresh games against the model (greedy decoding, with the inference constraints below) — so you watch solve rate climb, not just loss drop. Loss and solve rate do not move in lockstep; solve rate is the number that matters.

## Inference constraints: only legal moves

At generation time (`evaluate.py::next_guesses`), the model's logits are masked twice before sampling:

1. **Consistency filter.** Any word that contradicts the feedback already seen cannot be the secret, so it can never be a correct guess. Masked out.
2. **Repeat suppression.** Guesses already played are masked out. A repeated non-winning guess is always wasted.

Both masks only remove moves that are provably bad, so they never hurt the model's ceiling — they delete a class of silly losses the teacher never makes.

## Why a 0.2B model beats frontier LLMs at this

The API benchmark asks frontier chat models to do the same thing: conversation history in, one valid word out. They score 70-95% depending on the model. This 0.2B model, fully trained, scores 95-98%.

Not because it's smarter. Because the task is closed and the model was trained for exactly it:

- The LLM is doing general next-token prediction over all of language; Wordle is a rounding error in its training distribution. It has seen Wordle puzzles, but it has never been trained to optimize solve rate.
- The tiny model's entire 215M parameters and 7M training tokens are spent on one game. Its prior is Wordle-shaped.
- Frontier LLMs also get the consistency constraint removed by how they generate — a chat model that guesses a word contradicting its own feedback is just a token error; a bad guess. The small model's logits are masked so that class of error cannot happen.
- And there's no memorization loophole: the holdout answers were never in training, and the LLMs can't have memorized all 2,315 answer solutions either (they're not all in the training corpus).

So the honest framing: the small model wins this narrow benchmark because it is a specialist. The benchmark exists to prove that parameter count is not capability — task alignment is. That's the point of the repo.

## How to read the numbers

- Solve rate on the 200-word holdout is the only number that measures skill. Anything evaluated on training answers measures memory.
- At 300 evaluation games, ±2.5 points of noise. At 1000, ±1.4. Don't chase differences under 3 points.
- The teacher is your ceiling (~99%). The exact solver is the theoretical band above it (~100%, ~3.4 avg). Past ~99% solve rate, the interesting metric is average guesses.

## The upgrade path, in one paragraph

More data with more exploration (cheap), letter-level signal (medium), self-play distillation and policy gradients (the real ceiling-breakers), and MCTS with the model as policy heuristic (99-100% solve). Full detail, effort estimates, and expected gains in `docs/improvement_walkthrough.md`. The one trap: never tune on the training distribution — always report the holdout.
