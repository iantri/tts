#!/usr/bin/env python3
"""
Qwen3-TTS CLI for VoiceDesign and CustomVoice generation.

References:
- https://github.com/QwenLM/Qwen3-TTS
"""

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


DEFAULT_MODEL_IDS = {
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}


def pick_device(requested: str) -> str:
    req = requested.lower()
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    req = requested.lower()
    if req == "float16":
        return torch.float16
    if req == "float32":
        return torch.float32
    if req == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda") or device == "mps":
        return torch.bfloat16
    return torch.float32


def get_unique_filename(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    counter = 2

    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-TTS CLI for voice-design and custom-voice generation "
            "(voice cloning intentionally excluded)."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize. If omitted, stdin will be used when piped.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_kind",
        choices=["voice-design", "custom-voice"],
        default="voice-design",
        help="Model family to use.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="Auto",
        help="Target language (e.g. Auto, English, Chinese).",
    )
    parser.add_argument(
        "-i",
        "--instruct",
        default=None,
        help="Voice/style instruction. Required for voice-design.",
    )
    parser.add_argument(
        "-s",
        "--speaker",
        default=None,
        help="Speaker name for custom-voice (e.g. Vivian).",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print model-supported speakers and exit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection: auto|cuda:0|mps|cpu",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Dtype: auto|bfloat16|float16|float32",
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Optional attention implementation override, e.g. flash_attention_2.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress details while loading and generating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    text = args.text
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        elif not args.list_speakers:
            raise SystemExit(
                "No text provided. Pass text as an argument or pipe it via stdin."
            )

    if not args.list_speakers and not text:
        raise SystemExit("Text cannot be empty.")

    if args.model_kind == "voice-design" and not args.instruct and not args.list_speakers:
        raise SystemExit("--instruct is required with --model voice-design.")
    if args.model_kind == "custom-voice" and not args.speaker and not args.list_speakers:
        raise SystemExit("--speaker is required with --model custom-voice.")

    selected_model_id = args.model_id or DEFAULT_MODEL_IDS[args.model_kind]
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    log(f"[1/4] Using device={device}, dtype={dtype}")

    load_kwargs = {"device_map": device, "dtype": dtype}
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl

    log(f"[2/4] Loading model: {selected_model_id}")
    model = Qwen3TTSModel.from_pretrained(selected_model_id, **load_kwargs)

    if args.list_speakers:
        speakers = []
        if hasattr(model, "get_supported_speakers"):
            speakers = model.get_supported_speakers() or []
        if speakers:
            for speaker in speakers:
                print(speaker)
        else:
            print("No predefined speakers for this model.")
        return

    log("[3/4] Generating audio...")
    gen_start = time.time()
    if args.model_kind == "voice-design":
        wavs, sr = model.generate_voice_design(
            text=text,
            language=args.language,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
    log(f"[3/4] Generation finished in {time.time() - gen_start:.1f}s")

    output = Path(args.output)
    if args.output == "output.wav":
        output = get_unique_filename(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    log(f"[4/4] Writing WAV to: {output}")
    sf.write(str(output), wavs[0], sr)
    print(f"Saved audio to: {output} (total {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
