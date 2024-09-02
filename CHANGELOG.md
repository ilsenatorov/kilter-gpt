# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Deprecated
