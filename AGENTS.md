# Agent notes

KilterGPT trains a GPT that generates Kilterboard climbs. This repo only does data, training and evaluation; climbs are generated on-device by `../kilter-gpt-android` from an ONNX export.

- Setup: `uv sync`. Tests: `uv run python -m pytest`. Lint: `uvx ruff check kiltergpt scripts`.
- Whenever generation changes (`GPTModel.generate*`, `_generate_token`, tokenizer encoding of angle/grade, prompt validation, checkpoint loading), update the "Generating climbs" section of `README.md` in the same change. If the model's inputs change, note what the Android app and ONNX export need to change.
- beartype checks type hints at runtime across the whole `kiltergpt` package, so hints must match the values actually passed (angles are `int`).
- Lightning calls the datamodule's `setup()` again inside `fit`/`test`. Put any dataset filtering in `setup()`, not after it.
- `KilterDataset` reads rows by position (`iloc`). Bucket shuffling permutes index labels, so never look rows up by label.
