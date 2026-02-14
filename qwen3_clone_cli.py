#!/usr/bin/env python3
"""
Qwen3-TTS voice cloning CLI based on QwenLM/Qwen3-TTS examples.

Repository references:
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py
"""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


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
    # auto
    if device.startswith("cuda") or device == "mps":
        return torch.bfloat16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS voice cloning CLI (Base model only)."
    )
    parser.add_argument("text", help="Text to synthesize in cloned voice.")
    parser.add_argument(
        "-r",
        "--ref-audio",
        required=True,
        help="Reference audio path/URL/base64 supported by qwen-tts.",
    )
    parser.add_argument(
        "-t",
        "--ref-text",
        default=None,
        help="Transcript for reference audio. Optional if --x-vector-only is set.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="Auto",
        help="Target language (e.g. Auto, English, Chinese).",
    )
    parser.add_argument(
        "-o", "--output", default="clone.wav", help="Output WAV path."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model ID or local path.",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use speaker embedding only (ref-text not required; may reduce quality).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cuda:0|mps|cpu",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="auto|bfloat16|float16|float32",
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Attention impl override, e.g. flash_attention_2 or eager.",
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

    if not args.x_vector_only and not args.ref_text:
        raise SystemExit("--ref-text is required unless --x-vector-only is set.")

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    log(f"[1/4] Using device={device}, dtype={dtype}")

    load_kwargs = {
        "device_map": device,
        "dtype": dtype,
    }
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl

    log(f"[2/4] Loading model: {args.model}")
    model = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)

    log("[3/4] Running voice cloning generation...")
    gen_start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    log(f"[3/4] Generation finished in {time.time() - gen_start:.1f}s")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"[4/4] Writing WAV to: {output}")
    sf.write(str(output), wavs[0], sr)
    print(f"Saved cloned audio to: {output} (total {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
