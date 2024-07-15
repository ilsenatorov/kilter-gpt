# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Installation

`pip3 install torch torchvision torchaudio wandb lightning plotly pandas numpy matplotlib ipykernel jupyter beartype`

## Data

One thing you need to do is download the latest version of the kilterboard apk file, unzip it, and copy the `db.sqlite3` file to the `data` directory.
Then run the `preprocessing.ipynb` notebook to generate the data for training.

## Training

Simply running the `train.py` script will start training the model. You can adjust the hyperparameters in the script.

## TODO

* Improve tokenizer functionality, move all the tokenization logic to the tokenizer class
* Add some automatic evaluation metrics (similarity to real data, consistency, etc.)
* Better config handling
* Convert dataset to hf dataset?
* Add a script to convert the model to onnx?
* Plot UMAP of token embeddings to check for logical clustering
* Learning rate warmup with annealing is probably better than plateau reduction.
* ~~Save and load tokenizer from json/pickle~~
* ~~Improve consistency of data preprocessing store as pandas dataframe, handle internally as list/tensor~~
* ~~Validation set and loop?~~
* ~~Tokenize angle/difficulty and add them to the model~~
* ~~Log the model to wandb~~
* ~~Add a script/notebook to generate climbs from the model~~

## Permutation invariant issues

The model is currently not permutation invariant, meaning that the order of the input tokens matters.
This is not ideal for the task of generating climbing routes, where the order of the holds is not important.

## Design choices and reasonings

* GPT model is a simple baseline for text generation tasks, and it has been shown to work well for a variety of tasks.
* The model is trained on the text representation of the climbing routes, which is a sequence of holds. Here the issues arises with the permutation invariance of the model, as the order of the holds is not important. Currently I solve it by shuffling the input sequence, but this is not ideal.
* Tokenisation - there are currently 5 types of tokens - special tokens (bos, eos, pad etc), hold tokens (hold id), hold role (color), wall angle and grade. Each climb is then represented as alternating pair of hold id and color tokens, with angle and difficulty prepended. This keeps the vocabulary small and manageable.
* Every time I access a climb in the dataset, I cut out a random sequence of size `context_len` out of it. If the sequence is smaller I pad it on the left side. This is to simulate the model generating the climb one hold at a time.
