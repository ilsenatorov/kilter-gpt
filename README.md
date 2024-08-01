# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Installation

`pip3 install torch torchvision torchaudio wandb lightning plotly pandas numpy matplotlib ipykernel jupyter beartype`

## Data

One thing you need to do is download the latest version of the kilterboard apk file, unzip it, and copy the `db.sqlite3` file to the `data` directory.
Then run the `preprocessing.ipynb` notebook to generate the data for training.

## Training

Simply running the `train.py` script will start training the model.
You can adjust the hyperparameters in the script.
By default the dataset is constructed from `data/raw/climbs.csv` file that is generated from the preprocessing notebook.

## TODO

* Add some automatic evaluation metrics (similarity to real data, consistency, etc.)
* Add code to convert model to torchscript/onnx
* Add code to host the model as http API
* Try to add data from routes to the dataset
* Generation that ensures sensibility - no breaking limits.
* Tests for basic functionality, especially HTTP API
* Simple metrics for diversity/sensibility
* Hyperparameter tuning
* Add simple tests for the model
* ~~Learning rate warmup with annealing is probably better than plateau reduction.~~
* ~~Add masking of padding tokens to attention mechanism~~
* ~~Improve tokenizer functionality, move all the tokenization/padding logic to the tokenizer class~~
* ~~Save and load tokenizer from json/pickle~~
* ~~Improve consistency of data preprocessing store as pandas dataframe, handle internally as list/tensor~~
* ~~Validation set and loop?~~
* ~~Tokenize angle/difficulty and add them to the model~~
* ~~Log the model to wandb~~
* ~~Add a script/notebook to generate climbs from the model~~
* ~~Better config handling~~

## Permutation invariant issues

The model is currently not permutation invariant, meaning that the order of the input tokens matters.
This is not ideal for the task of generating climbing routes, where the order of the holds is not important.

## Design choices and reasonings

* GPT model is a simple baseline for text generation tasks.
* The model is trained on the text representation of the climbing routes, which is a sequence of holds. Here the issues arises with the permutation invariance of the model, as the order of the holds is not important.
* Shuffling - the order of holds is shuffled on every pass, which should allow prompts that consist only of start and finish holds to generate different routes.
* Label smoothing - if set to `True`, and the token to be predicted is a hold token, all of the remaining holds in a route will be accepted as valid answers. This is to prevent the model from overfitting to the exact order of holds.
* Tokenisation - there are currently 5 types of tokens - special tokens (bos, eos, pad etc), hold tokens (hold id), hold role (color), wall angle and grade. Each climb is then represented as alternating pair of hold id and color tokens, with angle and difficulty prepended. This keeps the vocabulary small and manageable.
* Every time I access a climb in the dataset, I cut out a random sequence of size `context_len` out of it. If the sequence is smaller I pad it on the left side. This is to simulate the model generating the climb one hold at a time.
