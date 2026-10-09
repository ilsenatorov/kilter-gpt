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
Run `uv run python scripts/preprocess.py` for the legacy database. For the downloaded 2026 layout-1 snapshot, run `uv run python scripts/preprocess.py --sqlite_path data/boardsesh-kilter-layout1-2026-10-03.db --out_dir data/processed-broad-2026 --broad`. This keeps zero-ascent climbs but excludes climbed routes rated below 2. The snapshot's quality averages mix the legacy 3-star and the newer 5-star ratings, so they are quantile-mapped onto the legacy 1–3 scale, fitted on climbs rated in both databases; a threshold then removes the same share of climbs from both sources. Matching reset climbs recover grades and ratings from `data/db.sqlite3`. Splits are fixed by hold/role pattern (90/9/1); `test-post-2024.csv` contains only held-out climbs absent from that legacy database. Existing `data/finetune` CSVs need regeneration before reuse.

## Training

Run `uv run python scripts/train.py --data_dir data/processed-broad-2026` to train on the broad dataset. Defaults use FP32 and 4096 tokens per batch on an RTX 2080 SUPER (FP16 skipped optimizer updates in the smoke test). For an offline smoke test: `WANDB_MODE=disabled uv run python scripts/train.py --data_dir data/processed-broad-2026 --subset 0.001 --epochs 1 --only_train true`. To compare local checkpoints on the same post-2024 holdout: `uv run python scripts/benchmark.py OLD.ckpt NEW.ckpt`.
Training drops the grade 10% of the time (`--p_uncond`) so generation can use classifier-free guidance: `generate_from_string(..., cfg_scale=2.0)` pushes climbs harder toward the requested grade (1.0 = plain conditional).

To measure grade fidelity, train the grade predictor on the same split: `uv run python scripts/train_grade.py`. Its `test/mae` on real climbs is the floor. Pass its checkpoint as `--grade_predictor` to `train.py` or `benchmark.py` to log `test/temp=T/cfg=W/grade_mae`, the mean absolute difference in difficulty units (1 unit ≈ one Font subgrade) between the requested grade and the predicted grade of valid generated climbs. `sweep.yaml` is a grid over model sizes ranked by this metric, using 30-epoch runs tested on a random 30% of the test set (`--test_subset`). It expects the checkpoint at `data/grade_predictor.ckpt`.

Two test metrics compare models whatever their tokenization. `test/set_nll` is the negative log-likelihood (nats per climb) of each held-out climb as an unordered set of holds, so lower is better. Hold-shuffled models are scored on 8 random orders (mean likelihood × n!), `--bottom_up` models on their single order. `test/temp=T/cfg=W/c2st_auc` is a classifier two-sample test: an MLP is cross-validated on telling generated climbs from the real test climbs they were prompted from, so 0.5 means the two are indistinguishable. Two real samples score 0.49; moving one random hold of each real climb scores 0.64, three holds 0.83. Each real climb and the climb generated from its prompt stay in the same fold, and a climb repeated at several angles counts once. `max_hand_gap` is the largest vertical gap between consecutive hand holds, as a fraction of board height. Test prompts are the first 20% of each climb's holds in CSV order, which is sorted by placement, so they are mostly low bolt-on holds. A hold-shuffled model reads its prompt as a random sample of the climb, so it then under-uses the screw-on holds (placements 1447+). Pass `--shuffle_test_prompts` to `benchmark.py` to prompt with a random subset of the holds instead; on the baseline checkpoint this drops `c2st_auc` from 0.61 to 0.53.

Experimental flags: `--bottom_up true` trains on holds listed bottom to top instead of in random order. Half the time a random prompt of up to 30% of the holds comes first, so prompts still work. `--hold_positions true` adds a learned function of each hold's board position (Fourier features of `figs/image_coords.csv`) to its tied input/output embedding. Neither flag changes the model's inputs or the generation code.

You can adjust the hyperparameters in the script.
By default the dataset is loaded from `data/processed` folder which contains the three csv files for train, val and test.
By defauly it's generated from the `python scripts/preprocess.py` script.

## Generating climbs

```python
from kiltergpt.models import GPTModel
from kiltergpt.utils import Plotter

model = GPTModel.load_checkpoint("path/to/model.ckpt")  # or GPTModel.load_from_wandb("ilsenatorov/kilter-gpt/model-<run_id>:v0")
frames = model.generate_from_string("p1387r14", angle=40, grade="7a")  # temperature 1, top-p 1, cfg_scale 3
image = Plotter().plot_climb(frames)  # RGB array of the board with the climb drawn on it
```

- `frames`: holds the climb must contain, as `p<placement>r<role>` pairs. Roles: `r12` start, `r13` hand, `r14` finish, `r15` foot. Use `""` to let the model pick every hold. Holds must be unique placements of the original layout (1073–1599, excluding 1396–1446). A prompt can have at most 2 starts, 2 finishes and 28 holds. An invalid prompt raises `ValueError`.
- `angle`: board angle in degrees (int).
- `grade`: a Font grade from `Tokenizer.grade_tokens()` (`"6b+"`, `"7a"`, …), or a float in Kilter difficulty units (16 = 6a/V3, 22 = 7a/V6).
- `temperature`, `p`: sampling temperature and top-p, both `1.0` by default. At these values generated climbs match real ones (about 12 holds, 4 of them feet). Lowering either one gives more conventional hand sequences but drops foot holds first and cuts climbs short: at `0.7` / `0.8` climbs average 8 holds and under 2 feet.
- `cfg_scale`: classifier-free guidance strength on the grade. `1.0` is plain conditional sampling; higher values follow the grade more closely and reduce variety. The default (`None`) uses `3.0` on checkpoints trained with `--p_uncond > 0` and `1.0` on older ones, including `kiltergpt.BEST_MODEL`, which raise `ValueError` for any explicit value other than `1.0`. To pick a value, run `scripts/benchmark.py CKPT --grade_predictor data/grade_predictor.ckpt --cfg_scales 1 2 3` and compare `grade_mae` against `novel_rate`.

- `ascents`: for checkpoints trained with `--popularity true`, generate climbs like ones with about this many ascents (`None`, the default, conditions on any popularity). Other checkpoints raise `ValueError` for anything but `None`.

The result keeps the prompt holds and always has 1–2 starts, 1–2 finishes and 4–28 holds. With a role-first checkpoint (`--role_first true`), temperature and top-p only shape which hold comes next; roles and the stop decision always sample at temperature 1, so lowering the temperature no longer drops foot holds. Generation always samples in eval mode without gradients, so you don't need to call `.eval()` first. `model.generate_generation_fig(prompt, angle, grade, ...)` also saves a GIF of the model's hold probabilities at each step.

On-device generation lives in `../kilter-gpt-android`. `uv run python scripts/export_onnx.py CKPT OUT.onnx` exports a checkpoint and checks the export against PyTorch. Its inputs are `tokens` (`[BOS]` = 1, then token ids), `angle` (`angle / 70`), `grade` (`(grade index + 1) / 39`) and, for checkpoints trained with `--p_uncond > 0`, `uncond` (one bool per row). The app's sampling loop has to mirror `GPTModel._generate_token`: the hold/role masks, top-p, and guidance, where the conditioned and `uncond` rows run as one batch of two. Its defaults should match `generate_from_string`. Checkpoints trained with `--popularity true` also take `popularity` (`(B, 1)` float, `log10(1 + ascents) / 4` capped at 1, see `Tokenizer.popularity`), passed to both guidance rows, and their logits have one more prefix position. Checkpoints trained with `--role_first true` emit each hold as role then placement (`r13`, `p1234`) instead of placement then role; for those the app has to swap the order in its prompt, masks and decoding, and sample roles and the stop token at temperature 1 as `_generate_token` does.

## TODO

- Add an option to select a predefined style for the climb
- ~~Add code to convert model to torchscript/onnx~~
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
