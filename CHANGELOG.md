# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Classifier-free guidance on the grade (`--p_uncond` in training, `cfg_scale` in generation).
- Grade predictor (`scripts/train_grade.py`) and `grade_mae` test metric.
- Test metrics `mean_holds`, `mean_feet` and `no_feet_rate` for generated climbs, next to `test/real/*` for the test set.
- `scripts/export_onnx.py` exports a checkpoint for the Android app, with the `uncond` input for guidance, and checks it against PyTorch.

### Changed

- Test-time generation samples at temperature 0.7 / top-p 0.8 (as before, for comparison) and at the new 1 / 1 default, with guidance scales 1 and 3. The sweep ranks by `test/temp=1.0/cfg=3.0/grade_mae`.
- `generate_from_string` now defaults to temperature 1, top-p 1 and `cfg_scale` 3 (1 on checkpoints without guidance). The old 0.7 / 0.8 defaults cut climbs to about 8 holds and dropped most foot holds.
- Snapshot quality averages are quantile-mapped to the legacy 1–3 scale before filtering.

### Removed

- The FastAPI/Supabase web service (`kiltergpt/app`, `__main__.py`, `GPTModel.get_fastapi_app`, `Hasher`); the model now runs on-device in the Android app.

### Fixed

- `DynamicBatchSampler` computed token budgets from the wrong rows after bucket shuffling.
- The train set was never shuffled (`KilterDataModule(shuffle=False)` by default), so every epoch walked the CSVs in ascent-count order.
- `DynamicBatchSampler` kept the length buckets contiguous after bucket shuffling, so an epoch walked one climb length at a time. Batch order is now shuffled too, and a too-long first climb no longer yields an empty batch.

## 0.4.0 - 2024-10-01

### Added

- Return entropy of a generation in the API.
- Plotter highlights the probabilities.

### Changed

- API overhaul, now uses supabase to store feedback.
- New BEST_MODEL.

### Removed

### Fixed

- Fixed the grade calculation in preprocessing and tokenization.

## [0.3.1] - 2024-09-13

### Added

- `BEST_MODEL` const in `__init__.py` to keep track of the best model.
- More verbose logging in `app.py`.

## [0.3.0] - 2024-09-07

### Added

- Added climb name hashing into adjective-noun pairs.
- Added finetuning script.

### Changed

- More fine-grained control over the preprocessing, allows for multiple datasets.
- Angle and grade are now continuous variables.
- Loss is no longer calculated for the angle and grade.
- FastAPI now also returns meta-data (prompt, angle, grade, name, version).
- FastAPI must run without any arguments.

## [0.2.0] - 2024-09-01

### Added

- Added a `CHANGELOG.md` file to keep track of changes in the project.
- Added DataModule tests.
- Added Plotter tests.
- Added lr scheduler tests.
- Added test_step tests.
- Added token distribution spearman correlation coefficient to the metrics.
- Added jaccard similarity to the metrics.
- `Tokenizer.onehot` now also works on encoded tensors.
- Choose data split in `preprocess.py`.
- Added highlighting to plotter.
- Added image logging at the end of each epoch.
- Set limit on the number of start and finish holds to 2 during generation.
- Set limit to 30 holds (64 tokens) during generation.
- Added batched shuffling.
- Sample sweep configuration.
- Added `DynamicBatchSampler` for dynamic batch sampling.
- Added bucket shuffling to `DynamicBatchSampler`.
- Added only_train flag to train.py that skips testing.
- Added label smoothing to option.
- Added gradient clipping
- Prevented repetition of holds in the same sequence.

### Changed

- Changed `utils.py` to `utils/` and created own files for each different utils.
- Removed tokens from 12x14 kilterboard from the dataset.
- Reverted to randomised dataset sampling.
- Renamed `KilterGPTDataset` to `KilterDataset`.
- No padding needed now!
- Moved training loop logic to a separate file.
- Loading from wandb is now done from model-registry by default.

### Removed

- Removed fused optimizer

### Fixed

- Fixed paths in `preprocess.py` to be OS agnostic.
- Fixed paths in other scripts to be OS agnostic.
- Removed hard-coded devices count in `train.py`.
- Fixed DataModule `num_workers` not being used.
