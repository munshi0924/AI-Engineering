"""
system_info.py

Gathers basic hardware/OS info so it can be embedded in the porting prompt.
Telling the model what CPU/OS it's targeting lets it make better choices
(e.g. SIMD width, native codegen flags) than a generic prompt would.
"""

import platform
import os


def retrieve_system_info() -> str:
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    return (
        f"OS: {info['os']} ({info['os_version']})\n"
        f"Architecture: {info['machine']}\n"
        f"Processor: {info['processor']}\n"
        f"CPU cores: {info['cpu_count']}"
    )


if __name__ == "__main__":
    print(retrieve_system_info())
