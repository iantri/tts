# Qwen3 TTS Tools

> Note: This project is fully vibecoded by Codex, so parts may or may not work as-is in your environment.

Two command-line tools for Qwen3 text-to-speech:

- `qwen3_tts_cli.py`: text-to-speech with `voice-design` and `custom-voice`
- `qwen3_clone_cli.py`: voice cloning from a reference WAV
- `qwen3_tui.py`: interactive text UI wizard that wraps both scripts

Included showcase sample:

- `prospector_cartoon2.wav`

## 1. Setup

Use Python 3.11+ (3.12 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

First run downloads model assets from Hugging Face.

## 2. NUMBA Cache Note (only if needed)

On some environments, `qwen-tts` dependencies can fail with:

`RuntimeError: cannot cache function '__o_fold'`

If you see that, prefix commands with:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache
```

If your machine runs without this, skip it.

## 3. CLI Reference: `qwen3_tts_cli.py`

Help:

```bash
python qwen3_tts_cli.py --help
```

### Positional input

- `text` (optional): text to synthesize
- If omitted, the tool reads from piped stdin

### Core options

- `-m, --model {voice-design,custom-voice}`: select model family
- `--model-id`: override model ID/path
- `-l, --language`: target language (default `Auto`)
- `-i, --instruct`: style instruction (required for `voice-design`)
- `-s, --speaker`: speaker name (required for `custom-voice`)
- `--list-speakers`: print predefined speakers and exit
- `-o, --output`: output WAV path (default `output.wav`)

### Generation controls

- `--top-k` (default `50`)
- `--top-p` (default `1.0`)
- `--temperature` (default `0.9`)
- `--repetition-penalty` (default `1.05`)
- `--max-new-tokens` (default `2048`)

### Runtime controls

- `--device`: `auto|cuda:0|mps|cpu`
- `--dtype`: `auto|bfloat16|float16|float32`
- `--attn-impl`: attention implementation override
- `-v, --verbose`: print load/generation timing

### Behavior notes

- With default output `output.wav`, the tool avoids overwrite by creating `output-2.wav`, `output-3.wav`, etc.
- For `voice-design`, `--instruct` is mandatory.
- For `custom-voice`, `--speaker` is mandatory.

### Examples

Voice-design:

```bash
python qwen3_tts_cli.py \
  --model voice-design \
  --language English \
  --instruct "Warm natural narration, mid-pitch, clear pacing." \
  --output output.wav \
  "Hello from Qwen3 voice design."
```

Custom-voice:

```bash
python qwen3_tts_cli.py \
  --model custom-voice \
  --speaker Vivian \
  --output custom_voice.wav \
  "This is a custom speaker test."
```

## 4. CLI Reference: `qwen3_clone_cli.py`

Help:

```bash
python qwen3_clone_cli.py --help
```

### Positional input

- `text` (required): text to synthesize in cloned voice

### Core options

- `-r, --ref-audio` (required): reference WAV path/URL/base64 supported by `qwen-tts`
- `-t, --ref-text`: transcript for reference audio
- `--x-vector-only`: embedding-only mode; allows omitting `--ref-text`
- `-l, --language`: target language (default `Auto`)
- `-o, --output`: output WAV path (default `clone.wav`)
- `--model`: model ID/path (default `Qwen/Qwen3-TTS-12Hz-1.7B-Base`)

### Generation + runtime controls

Supports the same `--top-k`, `--top-p`, `--temperature`, `--repetition-penalty`, `--max-new-tokens`, `--device`, `--dtype`, `--attn-impl`, and `-v` flags as `qwen3_tts_cli.py`.

### Behavior notes

- `--ref-text` is required unless `--x-vector-only` is set.
- Standard cloning quality is usually better with accurate `--ref-text`.

### Examples

Transcript-guided clone:

```bash
python qwen3_clone_cli.py \
  --ref-audio your_reference.wav \
  --ref-text "Transcript of the reference audio." \
  --output cloned.wav \
  "New sentence in the cloned voice."
```

Embedding-only clone:

```bash
python qwen3_clone_cli.py \
  --ref-audio your_reference.wav \
  --x-vector-only \
  --output cloned_xvector.wav \
  "Quick cloned sample without ref text."
```

## 5. Interactive TUI: `qwen3_tui.py`

Run the interactive wizard:

```bash
python qwen3_tui.py
```

What it does:

- Presents a menu for `voice-design`, `custom-voice`, or `voice clone`
- Prompts for required fields (text, output path, instruct/speaker/ref-audio/ref-text)
- Optionally prompts for advanced generation/runtime settings
- Shows the final command before execution and asks for confirmation
- Runs the underlying scripts (`qwen3_tts_cli.py` / `qwen3_clone_cli.py`)
- After successful generation, optionally plays audio via macOS `afplay`

By default, it sets `NUMBA_CACHE_DIR=/tmp/numba-cache` for launched commands.

Disable that behavior:

```bash
python qwen3_tui.py --no-numba-cache
```

## 6. Writing Better `--instruct` Prompts (Voice Control)

Based on Qwen3-TTS guidance from the official release/docs:

- Separate speaker identity from speaking style:
  - Identity: age range, gender presentation, accent, timbre
  - Style: mood, pacing, emphasis, rhythm, energy
- Be explicit and concrete:
  - Better: `elderly prospector, bright nasal twang, frontier drawl, energetic exclamations`
  - Worse: `funny voice`
- Include delivery constraints:
  - examples: `clear articulation`, `stable pacing`, `avoid distortion`, `natural pauses`
- Avoid contradictory instructions:
  - `very slow and very fast` in one prompt hurts consistency
- Iterate with small changes:
  - change one trait at a time (pitch, speed, emotion) to steer output predictably
- For multilingual or mixed-language text:
  - keep language expectations explicit in both `--language` and `--instruct`

Reusable template:

```text
[persona + age/context], [timbre + pitch + accent], [emotion + pacing + rhythm], [constraints for clarity/stability]
```

## 7. Reproduce Included Example

Command used to generate `prospector_cartoon2.wav`:

```bash
python qwen3_tts_cli.py \
  --model voice-design \
  --language English \
  --output prospector_cartoon2.wav \
  --instruct "Cartoon old-timey Wild West prospector, wiry elderly male voice, bright nasal twang with exaggerated frontier drawl, high-energy comedic shouting, big pitch jumps on excited words, dramatic pauses and gleeful yelps, crisp intelligible words without distortion." \
  "There's gold in these hills! Gold!! Yippeee!! Ho-ho! Strike up the banjo, we're rich by sunset!"
```

If needed:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache python qwen3_tts_cli.py ...
```

## 8. Output Check

```bash
ls -lh *.wav
file *.wav
```

## References

- Official blog URL requested: https://qwen.ai/blog?id=qwen3tts-0115
- Qwen3-TTS model usage docs and examples: https://www.alibabacloud.com/help/en/model-studio/qwen-tts
- Qwen3-TTS release overview mirror: https://www.alibabacloud.com/blog/602401
