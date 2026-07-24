from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from app.config import get_settings


def _install_frontend(frontend_directory: Path) -> None:
    if (frontend_directory / "node_modules").is_dir():
        return

    print("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=frontend_directory, check=True)


def _stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 5
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if process.poll() is None:
                process.kill()


def _wait_until_ready(processes: list[subprocess.Popen[bytes]]) -> None:
    deadline = time.monotonic() + 15
    ports = (8000, 5173)

    while time.monotonic() < deadline:
        failed_process = next(
            (process for process in processes if process.poll() is not None),
            None,
        )
        if failed_process is not None:
            raise RuntimeError(
                f"A dashboard process exited with status {failed_process.returncode}."
            )

        ready = True
        for port in ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    pass
            except OSError:
                ready = False
                break

        if ready:
            return
        time.sleep(0.1)

    raise RuntimeError("Dashboard startup timed out while waiting for local ports.")


def main() -> None:
    if shutil.which("npm") is None:
        raise SystemExit("Node.js and npm are required to run the React frontend.")

    settings = get_settings()
    _install_frontend(settings.frontend_directory)

    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--reload-dir",
        str(settings.project_root / "backend" / "src"),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    frontend_command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
    ]

    processes = [
        subprocess.Popen(api_command, cwd=settings.project_root),
        subprocess.Popen(frontend_command, cwd=settings.frontend_directory),
    ]

    interrupted = False
    startup_error: str | None = None
    try:
        _wait_until_ready(processes)
        print("Dashboard: http://127.0.0.1:5173", flush=True)
        print("API docs:  http://127.0.0.1:8000/docs", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
    except KeyboardInterrupt:
        interrupted = True
    except RuntimeError as error:
        startup_error = str(error)
    finally:
        _stop(processes)

    if interrupted:
        return

    if startup_error is not None:
        raise SystemExit(startup_error)

    failures = [
        process.returncode
        for process in processes
        if process.returncode not in (None, 0, -15)
    ]
    if failures:
        raise SystemExit(f"Dashboard process exited with status {failures[0]}")
