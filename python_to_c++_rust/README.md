# Polyglot Porter

Ports Python code to high-performance C++ **or** Rust using OpenAI models,
compiles it, and benchmarks it against the Python baseline for both speed
and correctness.

## Files

- `app.py` — everything: language config (C++/Rust), the 5-program test
  suite, prompt building, the porting call, the compile/run harness, the
  correctness checker, and the Gradio UI.
- `config.py` — OpenAI API key + client. `AVAILABLE_MODELS` lists which
  OpenAI models show up in the UI/CLI.
- `system_info.py` — gathers OS/CPU info to embed in the porting prompt.
- `benchmark.py` — CLI: runs every test program × every model × both
  languages, checks correctness, prints a speedup table. Imports its
  logic from `app.py`. Filterable with `--program`, `--model`, `--language`.
- `requirements.txt`

## Test suite (in `app.py`)

| Program | Pattern tested |
|---|---|
| Pi Approximation | Tight floating-point loop, no branching |
| Fibonacci Sum (mod) | Integer loop, tuple unpacking, modular arithmetic |
| Prime Sieve | Array allocation, nested loop, boolean indexing |
| Matrix Multiply | Triple-nested loop, 2D array/memory layout |
| Bubble Sort | O(n²) comparisons and swaps on a generated array |

All inputs are fixed (seeded, no live randomness) so Python, C++, and
Rust should produce matching results — checked automatically via the
`Result:` line each program prints.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

You need `g++` and `rustc` installed locally to compile the generated code.

## Run

```bash
python app.py                              # interactive UI
python benchmark.py                        # full grid: all programs x all models x both languages
python benchmark.py --program pi           # just one program
python benchmark.py --model gpt-5 --language cpp
```

## Status

The compile/run/correctness-check pipeline has been verified end-to-end
using hand-written reference C++ ports (matching Python's
`3.141592678590` result for the pi program). `benchmark.py` has been
confirmed to fail gracefully (not crash) when no API key is set.

The LLM-porting step itself hasn't been run against a live model yet —
add your `OPENAI_API_KEY` and it's ready to go.
