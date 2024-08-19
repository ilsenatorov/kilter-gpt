# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a `CHANGELOG.md` file to keep track of changes in the project.
- Added changelog-enforcer github action for pull request checks.
- Added DataModule tests.
- Added Plotter tests.

### Changed

### Removed

### Fixed

- Fixed paths in `preprocess.py` to be OS agnostic.
- Removed hard-coded devices count in `train.py`.
- Fixed paths in other scripts to be OS agnostic.
- Fixed DataModule `num_workers` not being used.

### Deprecated
