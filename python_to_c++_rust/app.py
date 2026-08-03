"""
app.py

Polyglot Porter — ports Python code to C++ or Rust using an OpenAI model,
compiles it, and compares it against the Python baseline for both speed
and correctness.

Everything except config.py, system_info.py, and benchmark.py lives here:
language config, the test-program suite, prompt building, the porting
call, the compile/run harness, and the correctness checker, followed by
the Gradio UI.
"""

import io
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass

import gradio as gr

from config import AVAILABLE_MODELS, REASONING_MODELS, get_client
from system_info import retrieve_system_info

WORK_DIR = "output"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# Each target language is one config entry: file name, compile command, run
# command, display name. Adding a third language means adding one entry
# here — nothing else in this file needs to change.

import platform

IS_WINDOWS = platform.system() == "Windows"
EXE_EXT = ".exe" if IS_WINDOWS else ""


def _exe_name(base: str) -> str:
    return f"{base}{EXE_EXT}"


@dataclass
class LanguageConfig:
    key: str
    display_name: str
    file_name: str
    compile_command: list[str] | None
    exe_name: str  # resolved to an absolute path at run time, not searched via PATH/cwd
    code_fence_tags: list[str]


LANGUAGES = {
    "cpp": LanguageConfig(
        key="cpp",
        display_name="C++",
        file_name="main.cpp",
        compile_command=(
            ["g++", "-std=c++17", "-O3"]
            + ([] if IS_WINDOWS else ["-march=native"])  # MinGW g++ often lacks native-arch support
            + ["-DNDEBUG", "main.cpp", "-o", _exe_name("main_cpp")]
        ),
        exe_name=_exe_name("main_cpp"),
        code_fence_tags=["cpp", "c++"],
    ),
    "rust": LanguageConfig(
        key="rust",
        display_name="Rust",
        file_name="main.rs",
        compile_command=(
            ["rustc", "-O"]
            + ([] if IS_WINDOWS else ["-C", "target-cpu=native"])
            + ["main.rs", "-o", _exe_name("main_rust")]
        ),
        exe_name=_exe_name("main_rust"),
        code_fence_tags=["rust", "rs"],
    ),
}


def get_language(key: str) -> LanguageConfig:
    if key not in LANGUAGES:
        raise ValueError(f"Unknown language '{key}'. Options: {list(LANGUAGES)}")
    return LANGUAGES[key]


# ---------------------------------------------------------------------------
# Test programs
# ---------------------------------------------------------------------------
# A small suite covering different computational patterns, so a model's
# performance on one doesn't stand in for all of them. Each is
# deterministic (fixed seeds/inputs) and prints a "Result:" line so
# Python and ported output can be compared directly.

@dataclass
class TestProgram:
    key: str
    name: str
    description: str
    code: str


PI_APPROXIMATION = TestProgram(
    key="pi",
    name="Pi Approximation",
    description="Leibniz-style series, tight floating point loop, no branching.",
    code='''
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(20_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip(),
)

FIBONACCI_SUM = TestProgram(
    key="fibonacci",
    name="Fibonacci Sum (mod)",
    description="Iterative integer loop with modular arithmetic, tests tuple unpacking translation.",
    code='''
import time

def calculate(n, mod):
    a, b = 0, 1
    total = 0
    for _ in range(n):
        a, b = b, (a + b) % mod
        total = (total + a) % mod
    return total

start_time = time.time()
result = calculate(20_000_000, 1_000_000_007)
end_time = time.time()

print(f"Result: {result}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip(),
)

PRIME_SIEVE = TestProgram(
    key="sieve",
    name="Prime Sieve",
    description="Sieve of Eratosthenes, tests array/list allocation and nested loop translation.",
    code='''
import time

def count_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    i = 2
    while i * i <= limit:
        if is_prime[i]:
            j = i * i
            while j <= limit:
                is_prime[j] = False
                j += i
        i += 1
    return sum(1 for p in is_prime if p)

start_time = time.time()
result = count_primes(5_000_000)
end_time = time.time()

print(f"Result: {result}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip(),
)

MATRIX_MULTIPLY = TestProgram(
    key="matmul",
    name="Matrix Multiply",
    description="Triple-nested loop over 2D arrays, tests memory layout / indexing translation.",
    code='''
import time

def multiply(n):
    A = [[(i * 3 + j * 7) % 13 for j in range(n)] for i in range(n)]
    B = [[(i * 5 + j * 11) % 17 for j in range(n)] for i in range(n)]
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return sum(sum(row) for row in C)

start_time = time.time()
result = multiply(300)
end_time = time.time()

print(f"Result: {result}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip(),
)

BUBBLE_SORT = TestProgram(
    key="sort",
    name="Bubble Sort",
    description="O(n^2) comparison sort on a deterministically generated array (LCG), tests swap/loop translation.",
    code='''
import time

def generate(n, seed):
    arr = []
    x = seed
    for _ in range(n):
        x = (1103515245 * x + 12345) % (2**31)
        arr.append(x % 1_000_000)
    return arr

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

start_time = time.time()
arr = generate(6000, 123456789)
arr = bubble_sort(arr)
result = sum(arr[:5]) + sum(arr[-5:])
end_time = time.time()

print(f"Result: {result}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip(),
)

PROGRAMS = {
    p.key: p for p in [PI_APPROXIMATION, FIBONACCI_SUM, PRIME_SIEVE, MATRIX_MULTIPLY, BUBBLE_SORT]
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def system_prompt_for(lang: LanguageConfig) -> str:
    return f"""
Your task is to convert Python code into high performance {lang.display_name} code.
Respond only with {lang.display_name} code. Do not provide any explanation other than occasional comments.
The {lang.display_name} response needs to produce identical output to the Python code, in the fastest possible time.
""".strip()


def user_prompt_for(python: str, lang: LanguageConfig, system_info: str) -> str:
    return f"""
Port this Python code to {lang.display_name} with the fastest possible implementation that produces identical output in the least time.

The system information is:
{system_info}

Your response will be written to a file called {lang.file_name} and then compiled and executed.
{"The compilation command is: " + " ".join(lang.compile_command) if lang.compile_command else "The code will be run directly."}

Respond only with {lang.display_name} code, and nothing else — no markdown fences, no explanation.

Python code to port:

```python
{python}
```
""".strip()


def messages_for(python: str, lang: LanguageConfig, system_info: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt_for(lang)},
        {"role": "user", "content": user_prompt_for(python, lang, system_info)},
    ]


# ---------------------------------------------------------------------------
# Porter
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str, lang: LanguageConfig) -> str:
    cleaned = text.strip()
    for tag in lang.code_fence_tags + [""]:
        fence = f"```{tag}"
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    if cleaned.rstrip().endswith("```"):
        cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def port(model: str, python_code: str, lang: LanguageConfig, output_dir: str = WORK_DIR) -> str:
    """Port python_code to `lang` using `model`. Returns the generated code
    and writes it to output_dir/lang.file_name."""
    client = get_client(model)
    system_info = retrieve_system_info()
    messages = messages_for(python_code, lang, system_info)

    kwargs = {"model": model, "messages": messages}
    if model in REASONING_MODELS:
        kwargs["reasoning_effort"] = "high"

    response = client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content
    reply = _strip_code_fences(reply, lang)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, lang.file_name)
    with open(out_path, "w") as f:
        f.write(reply)

    return reply


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_python(code: str) -> tuple[str, float]:
    """Runs Python code with stdout captured. Returns (output, elapsed_seconds)."""
    globals_dict = {"__builtins__": __builtins__}
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    start = time.time()
    try:
        exec(code, globals_dict)
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - start

    return output, elapsed


def compile_and_run(lang: LanguageConfig, work_dir: str = WORK_DIR, runs: int = 1) -> dict:
    """Compiles (if needed) and runs the generated code `runs` times.

    Returns {"success": bool, "outputs": [...], "timings": [...], "compile_error": str | None}
    """
    result = {"success": False, "outputs": [], "timings": [], "compile_error": None}

    if lang.compile_command:
        try:
            subprocess.run(
                lang.compile_command, cwd=work_dir,
                check=True, text=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            result["compile_error"] = e.stderr
            return result
        except OSError as e:
            # e.g. compiler not found on PATH (FileNotFoundError is an OSError)
            result["compile_error"] = (
                f"Could not run '{lang.compile_command[0]}': {e}\n"
                f"Is it installed and on your PATH?"
            )
            return result

    # Build an absolute path to the executable. On Windows, subprocess's
    # cwd= argument sets the *child's* working directory but the parent
    # process's own current directory is what gets searched to locate a
    # bare filename — so a relative name like "main_cpp.exe" can fail
    # with WinError 2 even though the file exists in work_dir. An
    # absolute path sidesteps that search entirely, on every OS.
    exe_path = os.path.abspath(os.path.join(work_dir, lang.exe_name))
    run_command = [exe_path]

    for _ in range(runs):
        start = time.time()
        try:
            proc = subprocess.run(
                run_command, cwd=work_dir,
                check=True, text=True, capture_output=True,
            )
            elapsed = time.time() - start
            result["outputs"].append(proc.stdout)
            result["timings"].append(elapsed)
        except subprocess.CalledProcessError as e:
            result["outputs"].append(f"Error: {e.stderr}")
            result["timings"].append(None)
            return result
        except OSError as e:
            # e.g. compiled binary not found (compile step didn't produce it)
            result["outputs"].append(
                f"Error: could not run '{exe_path}': {e}"
            )
            result["timings"].append(None)
            return result

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# Correctness comparison
# ---------------------------------------------------------------------------

def extract_result(output: str) -> str | None:
    match = re.search(r"^Result:\s*(.+)$", output, re.MULTILINE)
    return match.group(1).strip() if match else None


def results_match(python_output: str, ported_output: str, tolerance: float = 1e-6) -> bool:
    py_result = extract_result(python_output)
    ported_result = extract_result(ported_output)
    if py_result is None or ported_result is None:
        return False
    if py_result == ported_result:
        return True
    try:
        return abs(float(py_result) - float(ported_result)) < tolerance
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_KEY = "custom"
CUSTOM_PLACEHOLDER = '''
# Paste or write any Python code here, then use "Convert" below.
#
# If you want the correctness check (in "Compare") to work, make sure
# your code prints a line in this exact format:
#   print(f"Result: {some_value}")
# Any output before/after that line is fine — only that line gets compared.

import time

start_time = time.time()

# ... your code here ...
result = 42

end_time = time.time()
print(f"Result: {result}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''.strip()

PROGRAM_CHOICES = [(p.name, key) for key, p in PROGRAMS.items()] + [("Custom (write your own)", CUSTOM_KEY)]
DEFAULT_PROGRAM_KEY = "pi"


def load_program(program_key):
    if program_key == CUSTOM_KEY:
        return CUSTOM_PLACEHOLDER
    return PROGRAMS[program_key].code


def ui_convert(model, language_key, python_code):
    lang = get_language(language_key)
    try:
        code = port(model, python_code, lang, output_dir=WORK_DIR)
        return code, f"Ported to {lang.display_name} using {model}."
    except Exception as e:
        return "", f"Error: {e}"


def ui_run_python(python_code):
    """Runs just the Python side. Returns (display_text, state_dict_or_None)."""
    try:
        output, elapsed = run_python(python_code)
    except Exception as e:
        return f"Error running Python: {e}", None
    return f"{output}\n(took {elapsed:.4f}s)", {"output": output, "time": elapsed}


def ui_compile_run(language_key):
    """Compiles and runs just the ported code. Returns (display_text, state_dict_or_None)."""
    lang = get_language(language_key)
    try:
        result = compile_and_run(lang, work_dir=WORK_DIR, runs=1)
    except Exception as e:
        return f"Error compiling/running {lang.display_name}: {e}", None

    if not result["success"]:
        if result["compile_error"]:
            detail = result["compile_error"]
        elif result["outputs"]:
            detail = result["outputs"][-1]
        else:
            detail = "Unknown failure (no error captured)."
        return f"{lang.display_name} failed:\n{detail}", None

    output = result["outputs"][0]
    elapsed = result["timings"][0]
    return f"{output}\n(took {elapsed:.4f}s)", {"output": output, "time": elapsed, "lang": lang.display_name}


def ui_compare(py_state, lang_state):
    if not py_state:
        return "Run Python first."
    if not lang_state:
        return "Compile & run the target language first."

    py_output, py_time = py_state["output"], py_state["time"]
    lang_output, lang_time, lang_name = lang_state["output"], lang_state["time"], lang_state["lang"]

    py_result = extract_result(py_output)
    lang_result = extract_result(lang_output)

    if py_result is None or lang_result is None:
        correctness = "Correctness check skipped (no 'Result:' line found in one or both outputs)."
    elif results_match(py_output, lang_output):
        correctness = "Results match — correct."
    else:
        correctness = f"MISMATCH — Python gave '{py_result}', {lang_name} gave '{lang_result}'."

    speedup = py_time / lang_time if lang_time else None
    speed_line = f"{lang_name} was {speedup:.1f}x faster than Python ({lang_time:.4f}s vs {py_time:.4f}s)." if speedup else ""

    return f"{correctness}\n{speed_line}"


with gr.Blocks(title="Polyglot Porter") as ui:
    gr.Markdown("# Polyglot Porter — Python to C++ or Rust")

    with gr.Row():
        program_dropdown = gr.Dropdown(
            PROGRAM_CHOICES, label="Test program", value=DEFAULT_PROGRAM_KEY
        )
        load_btn = gr.Button("Load program")

    with gr.Row():
        python_box = gr.Textbox(label="Python code", lines=20, value=PROGRAMS[DEFAULT_PROGRAM_KEY].code)
        code_box = gr.Textbox(label="Generated code", lines=20)

    load_btn.click(load_program, inputs=[program_dropdown], outputs=[python_box])

    with gr.Row():
        model_dropdown = gr.Dropdown(
            AVAILABLE_MODELS, label="Model",
            value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None,
        )
        language_radio = gr.Radio(list(LANGUAGES.keys()), label="Target language", value="cpp")
        convert_btn = gr.Button("Convert")

    status_box = gr.Textbox(label="Status", interactive=False)
    convert_btn.click(ui_convert, inputs=[model_dropdown, language_radio, python_box], outputs=[code_box, status_box])

    gr.Markdown("## Run & compare")
    gr.Markdown(
        "Run each side independently, then compare. This lets you run Python once "
        "and try it against C++ and Rust separately without re-running Python each time."
    )

    # Holds the last result from each side, so Compare can use them even
    # though they were produced by separate button clicks.
    py_state = gr.State(None)
    lang_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            py_result_box = gr.Textbox(label="Python output", lines=6)
            run_python_btn = gr.Button("▶ Run Python")
        with gr.Column():
            lang_result_box = gr.Textbox(label="Compiled output", lines=6)
            compile_run_btn = gr.Button("⚙ Compile & Run")

    run_python_btn.click(ui_run_python, inputs=[python_box], outputs=[py_result_box, py_state])
    compile_run_btn.click(ui_compile_run, inputs=[language_radio], outputs=[lang_result_box, lang_state])

    compare_btn = gr.Button("Compare results")
    compare_result_box = gr.Textbox(label="Comparison", interactive=False)
    compare_btn.click(ui_compare, inputs=[py_state, lang_state], outputs=[compare_result_box])


if __name__ == "__main__":
    ui.launch(inbrowser=True)
