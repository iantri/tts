# Repository Guidelines

## Project Structure & Module Organization
- `qwen3_tts_cli.py`: direct text-to-speech CLI for `voice-design` and `custom-voice` models.
- `qwen3_clone_cli.py`: voice cloning CLI for the base model using reference audio.
- `README.md`: user-facing quickstart and command examples for both tools.
- `prospector_cartoon2.wav`: checked-in showcase output sample.
- `.gitignore`: excludes local caches/generated audio while allowing showcase sample.
- `__pycache__/`: Python bytecode cache (generated).

## Build, Test, and Development Commands
- `python qwen3_tts_cli.py --help`: show TTS CLI options.
- `python qwen3_tts_cli.py --model voice-design --instruct "warm narration" "Hello world"`: generate with instruction-driven voice.
- `python qwen3_tts_cli.py --model custom-voice --speaker Vivian "Test"`: generate with named custom speaker.
- `python qwen3_clone_cli.py --ref-audio your_reference.wav --ref-text "sample" "new line"`: cloning flow with transcript.
- `python qwen3_tts_cli.py --model voice-design --language English --output prospector_cartoon2.wav --instruct "Cartoon old-timey Wild West prospector, wiry elderly male voice, bright nasal twang with exaggerated frontier drawl, high-energy comedic shouting, big pitch jumps on excited words, dramatic pauses and gleeful yelps, crisp intelligible words without distortion." "There's gold in these hills! Gold!! Yippeee!! Ho-ho! Strike up the banjo, we're rich by sunset!"`: reproduce the checked-in showcase sample.
- If you hit a numba cache error, prefix commands with `NUMBA_CACHE_DIR=/tmp/numba-cache`.

## Coding Style & Naming Conventions
- Target Python 3.12+ and keep code compatible with the inline dependency block.
- Follow PEP 8: 4-space indentation, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Preserve argparse option patterns: long flags in kebab-case (for example `--ref-audio`), internal names in snake_case.
- Keep CLI errors explicit and actionable for invalid argument combinations.

## Testing Guidelines
- No formal test suite is currently committed; validate behavior with CLI smoke tests.
- Run `NUMBA_CACHE_DIR=/tmp/numba-cache python qwen3_tts_cli.py --help` as a baseline check.
- Run at least one generation path per touched mode (`voice-design`, `custom-voice`, and clone via `qwen3_clone_cli.py`).
- Verify each run creates a non-empty output audio file.
- If adding tests, use `pytest` under `tests/` with filenames like `test_<feature>.py`.

## Commit & Pull Request Guidelines
- Git history is not available in this workspace snapshot; use Conventional Commits by default (for example, `feat: add base-model ref-text validation`).
- Keep commits scoped to one logical change and include command evidence in PR descriptions.
- Include PR purpose and behavioral impact.
- Include reproduction/verification commands.
- Include sample CLI invocation for user-facing option changes.
- Link issue(s) when applicable.

## Security & Configuration Tips
- Do not commit private model credentials or environment-specific tokens.
- Keep cache and generated audio outputs out of commits unless intentionally adding fixtures.
