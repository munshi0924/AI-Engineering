"""
benchmark.py

CLI benchmark: for every test program, every model, every target language —
port, compile, run, check correctness, record speedup vs Python. Prints a
markdown table at the end. All the actual logic lives in app.py; this just
drives it without launching the Gradio UI.

Usage:
    python benchmark.py                # all programs, all models, both languages
    python benchmark.py --program pi
    python benchmark.py --model gpt-5
    python benchmark.py --language rust
"""

import argparse
import os

from config import AVAILABLE_MODELS
from app import (
    LANGUAGES, PROGRAMS, get_language,
    port, run_python, compile_and_run, results_match,
    WORK_DIR,
)


def run_benchmark(program_keys=None, model_names=None, language_keys=None):
    os.makedirs(WORK_DIR, exist_ok=True)
    program_keys = program_keys or list(PROGRAMS)
    model_names = model_names or AVAILABLE_MODELS
    language_keys = language_keys or list(LANGUAGES)

    rows = []
    for prog_key in program_keys:
        prog = PROGRAMS[prog_key]
        print(f"\n{'='*60}\n{prog.name}\n{'='*60}")

        print("Running Python baseline...")
        py_output, py_time = run_python(prog.code)
        print(py_output.strip())

        for model in model_names:
            for lang_key in language_keys:
                lang = get_language(lang_key)
                label = f"{prog.name} | {model} -> {lang.display_name}"
                print(f"\n{label}")

                try:
                    port(model, prog.code, lang, output_dir=WORK_DIR)
                except Exception as e:
                    print(f"  port failed: {e}")
                    rows.append((prog.name, model, lang.display_name, "port failed", "-", "-"))
                    continue

                result = compile_and_run(lang, work_dir=WORK_DIR, runs=1)
                if not result["success"]:
                    err = (result["compile_error"] or "run failed")[:150]
                    print(f"  {err}")
                    rows.append((prog.name, model, lang.display_name, "compile/run failed", "-", "-"))
                    continue

                lang_output = result["outputs"][0]
                lang_time = result["timings"][0]
                correct = results_match(py_output, lang_output)
                speedup = py_time / lang_time if lang_time else None

                status = "correct" if correct else "WRONG RESULT"
                print(f"  {status}, {lang_time:.6f}s, {speedup:.1f}x speedup" if speedup else f"  {status}")

                rows.append((
                    prog.name, model, lang.display_name,
                    status,
                    f"{lang_time:.6f}s",
                    f"{speedup:.1f}x" if speedup else "-",
                ))

    print("\n\n| Program | Model | Language | Status | Time | Speedup |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {' | '.join(row)} |")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", action="append", help="Program key(s): pi, fibonacci, sieve, matmul, sort")
    parser.add_argument("--model", action="append", help="Model name(s), e.g. gpt-5")
    parser.add_argument("--language", action="append", help="Language key(s): cpp, rust")
    args = parser.parse_args()

    run_benchmark(
        program_keys=args.program,
        model_names=args.model,
        language_keys=args.language,
    )
