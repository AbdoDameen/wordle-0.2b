# Benchmarks

The benchmark pokes every contestant with the same secret words, the same rules, and 6 guesses. Chat models get the game history as a conversation and must reply with a single valid word; the local model and the teacher play through the same game engine. Everything lands in `results/`.

## Running it

```bash
# every contestant you have keys for
python benchmarks/api_benchmark.py --models \
  openai/gpt-4o-mini,anthropic/claude-sonnet-4-5,deepseek/deepseek-chat,baseline \
  --games 30

# add your trained model
python benchmarks/api_benchmark.py --models baseline,local0.2b,openai/gpt-4o-mini \
  --checkpoint checkpoints/wordle-0.2b-final.pt --games 50 --plot
```

`--games 30` per model is a few cents on cheap models (gpt-4o-mini, deepseek-chat, gemini-flash) and a couple of dollars on the flagship ones. `--plot` renders a comparison chart next to the JSON/CSV results.

## Keys

The benchmark reads API keys from environment variables. Copy `.env.example` to `.env` and fill in what you have, or export them in your shell. `.env` is gitignored.

| Provider | Key env var | Format |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o-mini`, `openai/o3-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-5` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/<vendor>/<model>` |
| Together | `TOGETHER_API_KEY` | `together/<model>` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` (OpenAI-compatible endpoint) |
| Local (vLLM / Ollama / LM Studio) | `LOCAL_API_BASE` (+ optional `LOCAL_API_KEY`) | `local/your-model-name` |

No key needed: `baseline` (the entropy teacher) and `local0.2b` (your checkpoint, passed with `--checkpoint`).

## What gets measured

Per game: solved / lost / invalid (model never produced a legal word), guesses used, retries, latency, tokens. Rolled up: solve rate, average guesses, loss rate, invalid rate, average latency, estimated cost.

Costs come from a small pricing table in the script (`PRICING`). Prices drift; treat them as estimates. Unknown models report `None` cost.

## Protocol notes

- System prompt is identical for every model. It explains the rules, the G/Y/B feedback letters, and demands a single 5-letter word per reply.
- The conversation is the game history: each guess is an assistant message, each feedback a user message. No word list is given — the model has to know English words.
- Invalid or repeated replies get one correction nudge, then the game is scored `invalid` (a loss). This matters: frontier chat models sometimes insist on explaining instead of guessing, and the invalid rate is itself an interesting number.
- Temperature 0.7, `max_tokens` 24. The same seeds produce the same secrets for every contestant.
- Results vary by model version and provider routing. Re-run for your own numbers; don't trust a README from a stranger.

## Extending

New provider: add an entry to `OPENAI_COMPAT` (base URL + env var name). Anything OpenAI-compatible works, including local vLLM/Ollama/LM Studio servers. New pricing: add a row to `PRICING` (substring-matched).

To benchmark in a notebook instead of the CLI, `notebooks/03_benchmark_api_models.ipynb` does the same thing cell by cell.
