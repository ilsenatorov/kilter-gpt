# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Installation

`uv sync` creates and installs into `.venv` using the locked Python 3.11 dependencies.

Run tests with `uv run python -m pytest`.

## Data

The `db.sqlite3` file is downloaded with git LFS, so you need to have it installed to download the file.
Run `git lfs install` to install it, then `git lfs pull` to download the file.
The file contains the data for the climbs, which is then processed into a csv file for training.
If you want the updated version of the database, you can download it from the kitlerboard apk.
Run `uv run python scripts/preprocess.py` for the legacy database. For the downloaded 2026 layout-1 snapshot, run `uv run python scripts/preprocess.py --sqlite_path data/boardsesh-kilter-layout1-2026-10-03.db --out_dir data/processed-broad-2026 --broad`. This keeps zero-ascent climbs but excludes climbed routes rated below 2. Matching reset climbs recover grades and ratings from `data/db.sqlite3`. Splits are fixed by hold/role pattern (90/9/1); `test-post-2024.csv` contains only held-out climbs absent from that legacy database. Existing `data/finetune` CSVs need regeneration before reuse.

## Training

Run `uv run python scripts/train.py --data_dir data/processed-broad-2026` to train on the broad dataset. Defaults use FP32 and 4096 tokens per batch on an RTX 2080 SUPER (FP16 skipped optimizer updates in the smoke test). For an offline smoke test: `WANDB_MODE=disabled uv run python scripts/train.py --data_dir data/processed-broad-2026 --subset 0.001 --epochs 1 --only_train true`. To compare local checkpoints on the same post-2024 holdout: `uv run python scripts/benchmark.py OLD.ckpt NEW.ckpt`.
You can adjust the hyperparameters in the script.
By default the dataset is loaded from `data/processed` folder which contains the three csv files for train, val and test.
By defauly it's generated from the `python scripts/preprocess.py` script.

## TODO

- Add an option to select a predefined style for the climb
- Add code to convert model to torchscript/onnx
- Visualize the attention matrix
- Add perplexity
- Add benchmarking for the generation steps
- ~~Hyperparameter tuning~~
- ~~Check the block logic~~
- ~~Added batch sampling based on number of tokens~~
- ~~Write better tests~~
- ~~Improve the CI/CD side of things~~
- ~~Add some automatic evaluation metrics (similarity to real data, consistency, etc.)~~
- ~~Add simple tests for the model~~
- ~~Tests for basic functionality, especially HTTP API~~
- ~~Generation that ensures sensibility - no breaking limits.~~
- ~~Add code to host the model as http API~~
- ~~Learning rate warmup with annealing is probably better than plateau reduction.~~
- ~~Add masking of padding tokens to attention mechanism~~
- ~~Improve tokenizer functionality, move all the tokenization/padding logic to the tokenizer class~~
- ~~Save and load tokenizer from json/pickle~~
- ~~Improve consistency of data preprocessing store as pandas dataframe, handle internally as list/tensor~~
- ~~Validation set and loop?~~
- ~~Tokenize angle/difficulty and add them to the model~~
- ~~Log the model to wandb~~
- ~~Add a script/notebook to generate climbs from the model~~
- ~~Better config handling~~

## TOTRY

- Instead of tokenising grade and angle, treat them as continuous variables
- Write BERT-like model for clustering
- Use hold positions as extra information
- Use hand-crafted hold descriptors (pinch/jug/sloper/whatever)
- Try performer or other linear attention mechanisms
- ~~Use the normal GPT variant - with padding on the right and no slicing~~
