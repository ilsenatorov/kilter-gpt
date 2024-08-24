# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Installation

`pip3 install -r requirements.txt`

[Optional] If you want to install the package locally, you can run

`pip install -e .`

[Optional] Run tests with `pytest`

## Data

The `db.sqlite3` file is downloaded with git LFS, so you need to have it installed to download the file.
Run `git lfs install` to install it, then `git lfs pull` to download the file.
The file contains the data for the climbs, which is then processed into a csv file for training.
If you want the updated version of the database, you can download it from the kitlerboard apk.
Then run the `python scripts/preprocess.py` notebook to generate the data for training.

## Training

Simply running the `python scripts/train.py` script will start training the model.
You can adjust the hyperparameters in the script.
By default the dataset is loaded from `data/processed` folder which contains the three csv files for train, val and test.
By defauly it's generated from the `python scripts/preprocess.py` script.

## TODO

- Add code to convert model to torchscript/onnx
- Hyperparameter tuning
- Visualize the attention matrix
- Check the block logic
- Add perplexity
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

- Use the normal GPT variant - with padding on the right and no slicing
- Instead of tokenising grade and angle, treat them as continuous variables
- Write BERT-like model for clustering
- Use hold positions as extra information
- Use hand-crafted hold descriptors (pinch/jug/sloper/whatever)
-
