#!/usr/bin/env python3
"""
Interactive terminal UI for Qwen3-TTS scripts.

This wrapper calls:
- qwen3_tts_cli.py
- qwen3_clone_cli.py
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parent
TTS_SCRIPT = ROOT_DIR / "qwen3_tts_cli.py"
CLONE_SCRIPT = ROOT_DIR / "qwen3_clone_cli.py"


def prompt_text(label: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{label}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if not required:
            return ""
        print("This field is required.")


def prompt_choice(label: str, options: List[str], default: str) -> str:
    options_lc = {item.lower(): item for item in options}
    while True:
        joined = "/".join(options)
        raw = input(f"{label} ({joined}) [{default}]: ").strip()
        if not raw:
            return default
        if raw.lower() in options_lc:
            return options_lc[raw.lower()]
        print(f"Choose one of: {', '.join(options)}")


def prompt_yes_no(label: str, default: bool = True) -> bool:
    default_hint = "Y/n" if default else "y/N"
    raw = input(f"{label} [{default_hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer value.")


def prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Enter a numeric value.")


def maybe_prompt_advanced(args: List[str]) -> None:
    if not prompt_yes_no("Configure advanced generation/runtime settings?", default=False):
        return

    args.extend(["--device", prompt_text("Device", "auto")])
    args.extend(["--dtype", prompt_text("Dtype", "auto")])

    attn_impl = prompt_text(
        "Attention implementation (blank to skip, e.g. flash_attention_2)",
        default="",
    )
    if attn_impl:
        args.extend(["--attn-impl", attn_impl])

    args.extend(["--top-k", str(prompt_int("top-k", 50))])
    args.extend(["--top-p", str(prompt_float("top-p", 1.0))])
    args.extend(["--temperature", str(prompt_float("temperature", 0.9))])
    args.extend(["--repetition-penalty", str(prompt_float("repetition-penalty", 1.05))])
    args.extend(["--max-new-tokens", str(prompt_int("max-new-tokens", 2048))])

    if prompt_yes_no("Verbose mode?", default=False):
        args.append("-v")


def build_tts_command(mode: str) -> List[str]:
    if not TTS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {TTS_SCRIPT}")

    cmd = [sys.executable, str(TTS_SCRIPT), "--model", mode]
    cmd.extend(["--language", prompt_text("Language", "Auto")])
    cmd.extend(["--output", prompt_text("Output WAV path", "output.wav")])

    if mode == "voice-design":
        cmd.extend(["--instruct", prompt_text("Instruction prompt", required=True)])
    else:
        cmd.extend(["--speaker", prompt_text("Speaker name", required=True)])
        instruct = prompt_text("Optional instruction prompt", default="")
        if instruct:
            cmd.extend(["--instruct", instruct])

    maybe_prompt_advanced(cmd)
    cmd.append(prompt_text("Text to synthesize", required=True))
    return cmd


def build_clone_command() -> List[str]:
    if not CLONE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {CLONE_SCRIPT}")

    cmd = [sys.executable, str(CLONE_SCRIPT)]
    ref_audio = prompt_text("Reference WAV path", required=True)
    cmd.extend(["--ref-audio", ref_audio])
    cmd.extend(["--language", prompt_text("Language", "Auto")])
    cmd.extend(["--output", prompt_text("Output WAV path", "clone.wav")])

    use_x_vector = prompt_yes_no("Use x-vector-only mode?", default=False)
    if use_x_vector:
        cmd.append("--x-vector-only")
    else:
        cmd.extend(["--ref-text", prompt_text("Reference transcript", required=True)])

    maybe_prompt_advanced(cmd)
    cmd.append(prompt_text("Text to synthesize", required=True))
    return cmd


def run_command(cmd: List[str], use_numba_cache: bool) -> int:
    env = os.environ.copy()
    if use_numba_cache:
        env["NUMBA_CACHE_DIR"] = env.get("NUMBA_CACHE_DIR", "/tmp/numba-cache")

    print("\nCommand:")
    print(shlex.join(cmd))
    print("")
    if not prompt_yes_no("Run this command now?", default=True):
        print("Cancelled.")
        return 0

    try:
        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


def infer_output_path(cmd: List[str]) -> Path | None:
    if "--output" in cmd:
        idx = cmd.index("--output")
        if idx + 1 < len(cmd):
            return Path(cmd[idx + 1]).expanduser()
    if "-o" in cmd:
        idx = cmd.index("-o")
        if idx + 1 < len(cmd):
            return Path(cmd[idx + 1]).expanduser()
    return None


def resolve_generated_output(output_hint: Path | None) -> Path | None:
    if output_hint is None:
        return None

    resolved = output_hint.resolve()
    if resolved.exists():
        return resolved

    # qwen3_tts_cli.py may auto-rename output.wav to output-2.wav, output-3.wav, etc.
    parent = resolved.parent
    stem = resolved.stem
    suffix = resolved.suffix or ".wav"
    candidates = sorted(
        parent.glob(f"{stem}-*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def maybe_play_audio(output_hint: Path | None) -> None:
    if not prompt_yes_no("Play generated audio now (macOS afplay)?", default=False):
        return

    audio_path = resolve_generated_output(output_hint)
    if audio_path is None:
        manual = prompt_text("Could not locate output. Enter WAV path to play", required=False)
        if not manual:
            print("Skipped playback.")
            return
        audio_path = Path(manual).expanduser().resolve()

    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        return

    try:
        subprocess.run(["afplay", str(audio_path)], check=False)
    except FileNotFoundError:
        print("afplay is unavailable on this system.")


def print_menu() -> None:
    print("\nQwen3 TTS Interactive TUI")
    print("1) Voice Design (qwen3_tts_cli.py --model voice-design)")
    print("2) Custom Voice (qwen3_tts_cli.py --model custom-voice)")
    print("3) Voice Clone (qwen3_clone_cli.py)")
    print("4) Quit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive terminal wizard for qwen3_tts_cli.py and "
            "qwen3_clone_cli.py."
        )
    )
    parser.add_argument(
        "--no-numba-cache",
        action="store_true",
        help="Do not set NUMBA_CACHE_DIR for launched commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_numba_cache = not args.no_numba_cache

    print("This wizard wraps the existing CLI scripts.")
    if use_numba_cache:
        print("NUMBA_CACHE_DIR is enabled by default for compatibility.")
    else:
        print("NUMBA_CACHE_DIR is disabled.")

    while True:
        print_menu()
        selection = input("Select an option [1-4]: ").strip()
        if selection == "4":
            return 0
        if selection not in {"1", "2", "3"}:
            print("Invalid selection.")
            continue

        try:
            if selection == "1":
                cmd = build_tts_command("voice-design")
            elif selection == "2":
                cmd = build_tts_command("custom-voice")
            else:
                cmd = build_clone_command()
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        except KeyboardInterrupt:
            print("\nCancelled.")
            continue

        output_hint = infer_output_path(cmd)
        code = run_command(cmd, use_numba_cache=use_numba_cache)
        if code == 0:
            print("Done.")
            maybe_play_audio(output_hint)
        else:
            print(f"Command exited with code {code}.")

        if not prompt_yes_no("Run another task?", default=True):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
